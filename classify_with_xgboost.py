import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import joblib
import tensorflow as tf
import tensorflow_hub as hub

# --- Configuration (Same as before) ---
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# ------------------------------------

def load_movenet():
    return hub.load(MODEL_URL).signatures["serving_default"]

def load_image(path: Path):
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"\nWarning: Skipping {path} due to error: {e}")
        return None

def infer_keypoints(movenet_fn, img_np):
    img_resized = tf.image.resize_with_pad(tf.expand_dims(img_np, 0), INPUT_SIZE, INPUT_SIZE)
    img_tensor = tf.cast(img_resized, tf.int32)
    outputs = movenet_fn(img_tensor)
    return outputs["output_0"].numpy()[0, 0]

def gather_images(root: Path):
    files = []
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"Error: No class subfolders found in {root}.")
        return []
    for cls_dir in sorted(class_dirs):
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append((p, cls_dir.name))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with images sorted into class subfolders")
    ap.add_argument("--model", required=True, help="Path to your XGBoost .joblib model file")
    ap.add_argument("--out", required=True, help="Path to save the output predictions CSV")
    ap.add_argument("--threshold", type=float, default=0.2, help="Minimum average keypoint score")
    args = ap.parse_args()

    pack = joblib.load(args.model)
    model_pipeline = pack["model"]
    feature_names = pack["features"]
    label_encoder = pack["labels"] # ✅ Load the label encoder
    use_scores = pack.get("use_scores", False)

    movenet_fn = load_movenet()
    files = gather_images(Path(args.images))
    if not files: return

    rows = []
    for path, true_label in tqdm(files, desc="Classifying with XGBoost", unit="img"):
        img = load_image(path)
        if img is None: continue
            
        kps = infer_keypoints(movenet_fn, img)
        if np.mean(kps[:, 2]) < args.threshold: continue

        current_features = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            y, x, score = kps[i]
            if f"{name}_y" in feature_names: current_features[f"{name}_y"] = float(y)
            if f"{name}_x" in feature_names: current_features[f"{name}_x"] = float(x)
            if use_scores and f"{name}_score" in feature_names: current_features[f"{name}_score"] = float(score)

        X_live = pd.DataFrame([current_features], columns=feature_names)
        
        proba = model_pipeline.predict_proba(X_live)[0]
        pred_idx = np.argmax(proba)
        pred_label = label_encoder.inverse_transform([pred_idx])[0] # ✅ Use encoder to get text label
        confidence = float(proba[pred_idx])

        rows.append({
            "filename": str(path.as_posix()), "true_label": true_label,
            "predicted_label": pred_label, "confidence": confidence
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved predictions to: {args.out}")
    
    if not out_df.empty:
        acc = (out_df["true_label"] == out_df["predicted_label"]).mean()
        print(f"Accuracy on this folder: {acc*100:.2f}%")

if __name__ == "__main__":
    main()