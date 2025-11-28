import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import joblib
import tensorflow as tf
import tensorflow_hub as hub

# --- Configuration ---
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
# ---------------------

def load_movenet():
    # This function loads the TensorFlow Hub model.
    return hub.load(MODEL_URL).signatures["serving_default"]

def load_image(path: Path):
    # This function reads an image file and ensures it's in RGB format.
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Warning: Could not read image {path}. Skipping. Error: {e}")
        return None

def infer_keypoints(movenet_fn, img_np):
    # This function runs the MoveNet model on a single image.
    img_resized = tf.image.resize_with_pad(tf.expand_dims(img_np, 0), INPUT_SIZE, INPUT_SIZE)
    img_tensor = tf.cast(img_resized, tf.int32)
    outputs = movenet_fn(img_tensor)
    # The output shape is (1, 1, 17, 3) for y, x, score. We squeeze it.
    keypoints = outputs["output_0"].numpy()[0, 0]
    return keypoints

def gather_images(root: Path):
    # This function finds all image files in the class subdirectories.
    files = []
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"Error: No class subfolders found in {root}. Make sure your test set is organized by class.")
        return []
        
    for cls_dir in class_dirs:
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append((p, cls_dir.name))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with images sorted into class subfolders")
    ap.add_argument("--model", required=True, help="Path to your .joblib model file")
    ap.add_argument("--out", required=True, help="Path to save the output predictions CSV")
    ap.add_argument("--threshold", type=float, default=0.2, help="Minimum average keypoint score to process an image")
    args = ap.parse_args()

    # --- THE FIX STARTS HERE ---
    # 1. Load the model AND the feature names it was trained on.
    pack = joblib.load(args.model)
    clf = pack["model"]
    feature_names_from_training = pack["features"] # This is the crucial part!
    # --- END OF FIX PART 1 ---

    movenet_fn = load_movenet()
    files = gather_images(Path(args.images))
    if not files: return

    rows = []
    kept, skipped = 0, 0
    for path, true_label in tqdm(files, desc="Classifying images", unit="img"):
        img = load_image(path)
        if img is None:
            skipped += 1
            continue
            
        kps = infer_keypoints(movenet_fn, img)
        
        if float(np.mean(kps[:, 2])) < args.threshold:
            skipped += 1
            continue

        # --- THE FIX PART 2 ---
        # 2. Build a dictionary of features and then create a DataFrame
        #    using the EXACT column order from training.
        current_features = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            y, x, score = kps[i]
            # Add only the features the model was trained on
            if f"{name}_y" in feature_names_from_training:
                current_features[f"{name}_y"] = float(y)
            if f"{name}_x" in feature_names_from_training:
                current_features[f"{name}_x"] = float(x)
            if f"{name}_score" in feature_names_from_training:
                current_features[f"{name}_score"] = float(score)

        # Create a single-row DataFrame with the exact column order
        X_live = pd.DataFrame([current_features], columns=feature_names_from_training)
        # --- END OF FIX PART 2 ---

        # Now the prediction will work correctly
        proba = clf.predict_proba(X_live)[0]
        pred_idx = np.argmax(proba)
        pred_label = clf.classes_[pred_idx]
        confidence = float(proba[pred_idx])

        rows.append({
            "filename": str(path.as_posix()),
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence
        })
        kept += 1

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved predictions to: {args.out}")
    print(f"Processed: {kept}, Skipped: {skipped}")
    
    if not out_df.empty and "true_label" in out_df.columns:
        acc = (out_df["true_label"] == out_df["predicted_label"]).mean()
        print(f"Accuracy on this folder: {acc*100:.2f}%")

if __name__ == "__main__":
    main()