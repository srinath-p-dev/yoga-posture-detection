import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to test_data.csv")
    ap.add_argument("--model", required=True, help="Path to pose_cls_movenet.joblib")
    ap.add_argument("--out", default=None, help="Optional: predictions CSV")
    args = ap.parse_args()

    pack = joblib.load(args.model)
    clf = pack["model"]
    feat_cols = pack["features"]
    use_scores = pack.get("use_scores", False)

    df = pd.read_csv(args.csv)
    if not set(feat_cols).issubset(df.columns):
        raise ValueError("CSV does not contain the expected feature columns. Check your training features.")

    X = df[feat_cols]
    y = df["class_name"] if "class_name" in df.columns else None

    proba = clf.predict_proba(X)
    pred = np.array([clf.classes_[np.argmax(p)] for p in proba])
    conf = np.max(proba, axis=1)

    if y is not None:
        acc = accuracy_score(y, pred)
        print(f"\nAccuracy: {acc*100:.2f}%")
        print("\nClassification report:")
        print(classification_report(y, pred))
        print("Confusion matrix:")
        print(confusion_matrix(y, pred))

    if args.out:
        out_df = df.copy()
        out_df["predicted"] = pred
        out_df["confidence"] = conf
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nPredictions saved to: {args.out}")

if __name__ == "__main__":
    main()