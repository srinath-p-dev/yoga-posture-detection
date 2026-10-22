# live_coach_xgboost.py
# Real-time yoga pose classification + coaching (MoveNet + XGBoost/RF)
# Safe with pipelines that include StandardScaler (no NaNs by default).
# Enable masking only if your model can handle NaNs (e.g., XGBoost without scaler).

import argparse
import time
from collections import deque
from pathlib import Path
import threading

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import tensorflow_hub as hub
except Exception:
    hub = None

MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

EDGES = [
    ("nose","left_eye"),("nose","right_eye"),
    ("left_eye","left_ear"),("right_eye","right_ear"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),("left_knee","left_ankle"),
    ("right_hip","right_knee"),("right_knee","right_ankle")
]

def decode_class_names(pack, model):
    meta = pack.get("metadata", {})
    if "class_names" in meta:
        return list(meta["class_names"])
    le = pack.get("labels", None)
    try:
        if le is not None:
            return le.inverse_transform(getattr(model, "classes_", []))
    except Exception:
        pass
    classes = getattr(model, "classes_", None)
    if classes is None:
        return []
    return [str(x) for x in classes]

class MoveNetTFHub:
    def __init__(self, url=MODEL_URL, input_size=INPUT_SIZE):
        if hub is None:
            raise ImportError("tensorflow_hub not installed. Use --engine tflite or install TF Hub.")
        self.model = hub.load(url).signatures["serving_default"]
        self.input_size = input_size
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        _ = self.__call__(dummy)

    def __call__(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = tf.image.resize_with_pad(tf.expand_dims(img_rgb, 0), self.input_size, self.input_size)
        img_tensor = tf.cast(img_resized, tf.int32)
        outputs = self.model(img_tensor)
        kps = outputs["output_0"].numpy()[0, 0]  # [17,3] (y,x,score)
        return kps

class MoveNetTFLite:
    def __init__(self, model_path="movenet_thunder.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]
        self.input_size = int(self.in_details["shape"][1])
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        _ = self.__call__(dummy)

    def __call__(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = tf.image.resize_with_pad(tf.expand_dims(img_rgb, 0), self.input_size, self.input_size)
        tensor = img_resized.numpy().astype(self.in_details["dtype"])
        self.interpreter.set_tensor(self.in_details["index"], tensor)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_details["index"])
        return out[0, 0]  # [17,3]

def features_from_keypoints(kps, feature_names, kp_min=-1.0, use_scores=False):
    # kp_min < 0 => no masking (keeps all coords)
    feats = {name: np.nan for name in feature_names}
    for i, name in enumerate(KEYPOINT_NAMES):
        y, x, score = kps[i]
        use_val = (score >= kp_min) if kp_min >= 0 else True
        yv = float(y) if use_val else np.nan
        xv = float(x) if use_val else np.nan
        if f"{name}_y" in feats: feats[f"{name}_y"] = yv
        if f"{name}_x" in feats: feats[f"{name}_x"] = xv
        if use_scores and f"{name}_score" in feats: feats[f"{name}_score"] = float(score)
    return feats

def topk_from_proba(proba, class_names, k=3):
    idx = np.argsort(proba)[::-1][:k]
    return [(class_names[i], float(proba[i])) for i in idx]

class ProbaSmoother:
    def __init__(self, n=8):
        self.buf = deque(maxlen=int(max(1, n)))
    def update(self, p):
        self.buf.append(np.asarray(p, dtype=np.float32))
        return np.mean(self.buf, axis=0)

class StableClassifier:
    def __init__(self, class_names, window=12, min_ratio=0.6, min_conf=0.55):
        self.class_names = class_names
        self.labels = deque(maxlen=int(max(1, window)))
        self.probas = deque(maxlen=int(max(1, window)))
        self.min_ratio = min_ratio
        self.min_conf = min_conf
        self.current = None
        self.since = time.time()
    def update(self, proba):
        top_idx = int(np.argmax(proba))
        self.labels.append(top_idx)
        self.probas.append(proba)
        arr = np.array(self.labels)
        vals, cnts = np.unique(arr, return_counts=True)
        maj_idx = int(vals[np.argmax(cnts)])
        maj_ratio = cnts.max() / len(arr)
        avg_p = np.mean(self.probas, axis=0)
        maj_conf = float(avg_p[maj_idx])
        stable = (maj_ratio >= self.min_ratio) and (maj_conf >= self.min_conf)
        if stable:
            if self.current != maj_idx:
                self.current = maj_idx
                self.since = time.time()
        else:
            self.current = None
            self.since = time.time()
        return {
            "stable": stable,
            "label": self.class_names[maj_idx],
            "label_idx": maj_idx,
            "confidence": maj_conf,
            "hold_time": time.time() - self.since if stable else 0.0
        }

def draw_skeleton(frame, kps, kp_thresh=0.3, color=(46, 204, 113)):
    h, w = frame.shape[:2]
    pts = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        y, x, score = kps[i]
        if score < kp_thresh: continue
        px, py = int(x * w), int(y * h)
        pts[name] = (px, py)
        cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)
    for a, b in EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2)
    return frame

def compute_angle(a, b, c):
    a, b, c = np.array(a, np.float32), np.array(b, np.float32), np.array(c, np.float32)
    v1, v2 = a - b, c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosang = np.clip(np.dot(v1, v2) / denom, -1, 1)
    return float(np.degrees(np.arccos(cosang)))

def generic_coach_cues(kps, kp_thresh=0.3):
    cues = []
    idx = {n:i for i,n in enumerate(KEYPOINT_NAMES)}
    def ok(n): return kps[idx[n]][2] >= kp_thresh
    # Shoulders level
    if ok("left_shoulder") and ok("right_shoulder"):
        if abs(kps[idx["left_shoulder"]][0] - kps[idx["right_shoulder"]][0]) > 0.05:
            cues.append("Level your shoulders")
    # Arms straight-ish
    for side in ["left","right"]:
        sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        if ok(sh) and ok(el) and ok(wr):
            ang = compute_angle(kps[idx[sh]][:2], kps[idx[el]][:2], kps[idx[wr]][:2])
            if ang < 160:
                cues.append(f"Straighten your {side} arm")
    # Hips level
    if ok("left_hip") and ok("right_hip"):
        if abs(kps[idx["left_hip"]][0] - kps[idx["right_hip"]][0]) > 0.06:
            cues.append("Level your hips")
    return cues[:3]

class Speaker:
    def __init__(self, enabled=False, min_interval=4.0):
        self.enabled = enabled
        self.min_interval = float(min_interval)
        self.last_t = 0
        self._engine = None
        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception:
                print("TTS not available. Continuing without voice.")
                self.enabled = False
    def say(self, text):
        if not self.enabled or not text: return
        now = time.time()
        if now - self.last_t < self.min_interval: return
        self.last_t = now
        def _s():
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_s, daemon=True).start()

def main():
    ap = argparse.ArgumentParser(description="Live yoga coach with MoveNet + XGBoost/RF")
    ap.add_argument("--model", required=True, help="Path to .joblib pack (XGBoost/RF)")
    ap.add_argument("--engine", choices=["tflite","tfhub"], default="tflite")
    ap.add_argument("--tflite", default="movenet_thunder.tflite", help="Path to MoveNet TFLite")
    ap.add_argument("--source", default="0", help="Camera index or video path (e.g., 0 or video.mp4)")
    ap.add_argument("--mirror", action="store_true", help="Mirror webcam")
    ap.add_argument("--kp_threshold", type=float, default=0.2, help="Keypoint score threshold")
    ap.add_argument("--mask_low_conf", action="store_true",
                    help="Mask low-confidence joints as NaN (only enable if your model handles NaNs; default off)")
    ap.add_argument("--draw_kp_threshold", type=float, default=0.3)
    ap.add_argument("--smooth", type=int, default=8)
    ap.add_argument("--stable_window", type=int, default=12)
    ap.add_argument("--stable_ratio", type=float, default=0.6)
    ap.add_argument("--stable_conf", type=float, default=0.55)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--fps_limit", type=float, default=0.0)
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--speak_interval", type=float, default=4.0)
    args = ap.parse_args()

    # Video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    # Load classifier pack
    pack = joblib.load(args.model)
    model = pack["model"]
    feature_names = pack["features"]
    le = pack.get("labels", None)
    use_scores = pack.get("use_scores", False)
    class_names = list(decode_class_names(pack, model))

    # Backend
    movenet = MoveNetTFLite(args.tflite) if args.engine == "tflite" else MoveNetTFHub(MODEL_URL, INPUT_SIZE)

    smoother = ProbaSmoother(args.smooth)
    stabilizer = StableClassifier(class_names, args.stable_window, args.stable_ratio, args.stable_conf)
    speaker = Speaker(args.speak, args.speak_interval)

    dt_target = 1.0 / args.fps_limit if args.fps_limit and args.fps_limit > 0 else 0.0
    last_time = 0.0

    white = (255,255,255); green=(46,204,113); yellow=(241,196,15); blue=(52,152,219); gray=(180,180,180)

    while True:
        if dt_target > 0:
            now = time.time()
            if now - last_time < dt_target:
                time.sleep(max(0, dt_target - (now - last_time)))
            last_time = time.time()

        ok, frame = cap.read()
        if not ok: break
        if args.mirror and isinstance(src, int):
            frame = cv2.flip(frame, 1)

        kps = movenet(frame)
        draw_skeleton(frame, kps, args.draw_kp_threshold, color=green)

        # Build features (mask NaNs only if requested)
        kp_min = args.kp_threshold if args.mask_low_conf else -1.0
        feats = features_from_keypoints(kps, feature_names, kp_min=kp_min, use_scores=use_scores)
        X_live = pd.DataFrame([feats], columns=feature_names)

        proba = model.predict_proba(X_live)[0]
        proba_s = smoother.update(proba)
        topk = topk_from_proba(proba_s, class_names, k=max(1, args.topk))
        stab = stabilizer.update(proba_s)

        # UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10,10), (int(w*0.65), 160), (0,0,0), -1)
        cv2.putText(frame, "Yoga Live Coach", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)

        t1_lbl, t1_p = topk[0]
        tcolor = green if stab["stable"] else yellow
        cv2.putText(frame, f"Pred: {t1_lbl} ({t1_p*100:.1f}%)", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcolor, 2, cv2.LINE_AA)
        if stab["stable"]:
            cv2.putText(frame, f"Hold: {stab['hold_time']:.1f}s", (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2, cv2.LINE_AA)

        y0 = 125
        for i, (lbl, p) in enumerate(topk[1:], start=1):
            cv2.putText(frame, f"{i+1}. {lbl}: {p*100:.1f}%", (20, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gray, 2, cv2.LINE_AA)

        cues = generic_coach_cues(kps, kp_thresh=args.kp_threshold)
        if cues:
            cv2.rectangle(frame, (10, 160), (int(w*0.65), 160 + 25*(len(cues)+1)), (0,0,0), -1)
            cv2.putText(frame, "Coach:", (20,185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2, cv2.LINE_AA)
            for i, c in enumerate(cues):
                cv2.putText(frame, f"- {c}", (20, 185 + (i+1)*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2, cv2.LINE_AA)

        if stab["stable"]:
            speaker.say(f"Hold {stab['label']}")
        elif cues:
            speaker.say(cues[0])

        cv2.imshow("Live Coach (q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()