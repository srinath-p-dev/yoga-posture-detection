import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def build_feature_columns(use_scores=False):
    cols = []
    for name in KEYPOINT_NAMES:
        cols.extend([f"{name}_y", f"{name}_x"])
        if use_scores:
            cols.append(f"{name}_score")
    return cols

def main():
    ap = argparse.ArgumentParser(description="Train an XGBoost classifier on MoveNet CSV data.")
    ap.add_argument("--train_csv", required=True, help="Path to the training CSV file")
    ap.add_argument("--test_csv", required=True, help="Path to the testing CSV file")
    ap.add_argument("--out_model", default="pose_cls_xgboost.joblib", help="Path to save the output model file")
    ap.add_argument("--use_scores", action="store_true", help="Include keypoint scores as features.")
    args = ap.parse_args()

    feature_columns = build_feature_columns(use_scores=args.use_scores)
    
    print(f"--- Training XGBoost with {len(feature_columns)} features per sample ---")
    if args.use_scores: print("(Including keypoint scores)")

    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    # XGBoost requires integer labels (0, 1, 2...)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(df_train["class_name"])
    y_test_encoded = label_encoder.transform(df_test["class_name"])
    
    X_train = df_train[feature_columns]
    X_test = df_test[feature_columns]

    # Define the model pipeline
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print("\nTraining XGBoost model...")
    model_pipeline.fit(X_train, y_train_encoded)
    print("✅ Training complete.")

    print("\nEvaluating on test data...")
    y_pred_encoded = model_pipeline.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    accuracy = accuracy_score(df_test["class_name"], y_pred)

    print(f"\n🎯 Accuracy on test set: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(df_test["class_name"], y_pred, zero_division=0))

    model_package = {
        "model": model_pipeline,
        "features": feature_columns,
        "labels": label_encoder, # Save the label encoder as well!
        "use_scores": args.use_scores
    }
    joblib.dump(model_package, args.out_model)
    print(f"\n💾 Model saved to: {args.out_model}")

if __name__ == "__main__":
    main()