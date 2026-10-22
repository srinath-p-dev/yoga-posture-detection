import os, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
]

def process_image_folder(images_root, csv_out_path, min_avg_score=0.2):
    images_root = Path(images_root)
    if not images_root.is_dir():
        print(f"Error: Folder not found: {images_root}")
        sys.exit(1)
    
    movenet_fn = hub.load(MODEL_URL).signatures["serving_default"]

    header = ["filename"] + [f"{n}_{c}" for n in KEYPOINT_NAMES for c in ["y", "x", "score"]] + ["class_name"]
    rows = []

    for cls_dir in sorted([d for d in images_root.iterdir() if d.is_dir()]):
        for img_path in tqdm(cls_dir.rglob("*"), desc=f"Processing {cls_dir.name}", unit="img"):
            if img_path.suffix.lower() not in IMG_EXTS: continue
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                img_r = tf.image.resize_with_pad(tf.expand_dims(img, 0), INPUT_SIZE, INPUT_SIZE)
                kps = movenet_fn(tf.cast(img_r, tf.int32))["output_0"].numpy()[0,0]
                if np.mean(kps[:,2]) >= min_avg_score:
                    rows.append([str(img_path.as_posix())] + kps.flatten().tolist() + [cls_dir.name])
            except Exception as e:
                print(f"Skip {img_path}: {e}")

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(csv_out_path, index=False)
    print(f"\nSaved dataset to: {csv_out_path} ({len(rows)} images)")

if __name__ == "__main__":
    process_image_folder(images_root="yoga_poses/train", csv_out_path="train_data_new.csv")
    process_image_folder(images_root="yoga_poses/test", csv_out_path="test_data_new.csv")
    print("\n✅ Datasets re-created.")