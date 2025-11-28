import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def build_feature_columns(use_scores=False):
    """Builds a list of feature column names based on whether to include scores."""
    cols = []
    for name in KEYPOINT_NAMES:
        cols.extend([f"{name}_y", f"{name}_x"])
        if use_scores:
            cols.append(f"{name}_score")
    return cols

def main():
    ap = argparse.ArgumentParser(description="Train a RandomForest classifier on MoveNet CSV data.")
    ap.add_argument("--train_csv", required=True, help="Path to the training CSV file (e.g., train_data_new.csv)")
    ap.add_argument("--test_csv", required=True, help="Path to the testing CSV file (e.g., test_data_new.csv)")
    ap.add_argument("--out_model", default="pose_cls_movenet.joblib", help="Path to save the output model file")
    # This line adds the --use_scores flag
    ap.add_argument("--use_scores", action="store_true", help="If set, includes keypoint scores as features for training.")
    args = ap.parse_args()

    # Build the list of columns to use as features
    feature_columns = build_feature_columns(use_scores=args.use_scores)
    
    print(f"--- Training with {len(feature_columns)} features per sample ---")
    if args.use_scores:
        print("(Including keypoint scores)")
    else:
        print("(Using only y, x coordinates)")

    # Load data
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    # Prepare training and testing sets
    X_train = df_train[feature_columns]
    y_train = df_train["class_name"]
    X_test = df_test[feature_columns]
    y_test = df_test["class_name"]

    # Define the model pipeline
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Scales features to have zero mean and unit variance
        ("rf_classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Train the model
    print("\nTraining model...")
    model_pipeline.fit(X_train, y_train)
    print("✅ Training complete.")

    # Evaluate the model
    print("\nEvaluating on test data...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Accuracy on test set: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the trained model and its metadata
    model_package = {
        "model": model_pipeline,
        "features": feature_columns,  # Save the exact list of features used
        "use_scores": args.use_scores
    }
    joblib.dump(model_package, args.out_model)
    print(f"\n💾 Model saved to: {args.out_model}")

if __name__ == "__main__":
    main()