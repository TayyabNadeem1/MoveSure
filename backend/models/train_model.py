"""
MoveSure Model Training Script
Trains, evaluates, and saves the posture classification machine learning model.
Supports training on both foundational biomechanical datasets and user-collected datasets.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

from backend.models.feature_extractor import PostureFeatureExtractor
from backend.models.dataset_generator import generate_posture_samples, CLASS_LABELS


def train_posture_model(data_path=None, model_type="sgd", test_size=0.2, save_path="backend/models/posture_model.joblib"):
    print("=" * 60)
    print(" MoveSure Machine Learning Model Training")
    print("=" * 60)

    # 1. Load or Generate Dataset
    if data_path and os.path.exists(data_path):
        print(f"[*] Loading training data from: {data_path}")
        df = pd.read_csv(data_path)
        feature_cols = PostureFeatureExtractor.FEATURE_NAMES
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int32)
    else:
        print("[*] Generating biomechanical posture dataset...")
        X, y = generate_posture_samples(num_samples_per_class=800)

        # If user dataset exists, combine with baseline data
        user_data_path = "data/user_posture_dataset.csv"
        if os.path.exists(user_data_path):
            print(f"[*] Merging continuous user training samples from: {user_data_path}")
            user_df = pd.read_csv(user_data_path)
            feature_cols = PostureFeatureExtractor.FEATURE_NAMES
            X_user = user_df[feature_cols].values.astype(np.float32)
            y_user = user_df["label"].values.astype(np.int32)
            X = np.vstack([X, X_user])
            y = np.concatenate([y, y_user])

    print(f"[*] Total dataset size: {len(X)} samples across {len(np.unique(y))} classes.")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[*] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Selection & Training
    if model_type == "rf":
        print("[*] Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("[*] Training Incremental SGD Classifier (supports continuous online learning)...")
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            random_state=42
        )

    model.fit(X_train_scaled, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f"[*] Model Evaluation Accuracy: {acc * 100:.2f}%")
    print("=" * 60)
    target_names = [CLASS_LABELS[i] for i in sorted(np.unique(y))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 6. Save Model Artifacts
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, save_path)
    print(f"\n[+] Trained model saved to: {save_path}")

    return model, scaler, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoveSure Posture ML Model")
    parser.add_argument("--data", type=str, default=None, help="Path to custom CSV training dataset")
    parser.add_argument("--model", type=str, choices=["sgd", "rf"], default="sgd", help="Model type: sgd (online learner) or rf (Random Forest)")
    parser.add_argument("--output", type=str, default="backend/models/posture_model.joblib", help="Output path for trained model")
    args = parser.parse_args()

    train_posture_model(data_path=args.data, model_type=args.model, save_path=args.output)
