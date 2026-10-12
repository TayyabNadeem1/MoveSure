"""
MoveSure Root Training Shortcut
Allows users to train or retrain the posture model directly:
Usage: python train.py [--model sgd|rf] [--data path/to/dataset.csv]
"""

from backend.models.train_model import train_posture_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoveSure Posture Classifier")
    parser.add_argument("--data", type=str, default=None, help="Custom dataset CSV path")
    parser.add_argument("--model", type=str, choices=["sgd", "rf"], default="sgd", help="Model type (sgd or rf)")
    parser.add_argument("--output", type=str, default="backend/models/posture_model.joblib", help="Output model path")
    args = parser.parse_args()

    train_posture_model(data_path=args.data, model_type=args.model, save_path=args.output)
