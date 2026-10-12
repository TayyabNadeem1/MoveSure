"""
MoveSure Machine Learning Posture Classifier & Adaptive Calibration Engine
Replaces hardcoded thresholds with a trained Machine Learning model.
Learns from each user session and supports continuous model retraining.
"""

import os
import json
import time
import numpy as np
from backend.models.feature_extractor import PostureFeatureExtractor

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    SGDClassifier = None
    StandardScaler = None
    RandomForestClassifier = None


CLASS_LABELS = {
    0: "Good Posture",
    1: "Slouching",
    2: "Forward Head",
    3: "Uneven Shoulders",
}

FEEDBACK_MESSAGES = {
    0: "Good posture! Keep it up.",
    1: "Slouching detected! Sit upright and engage core.",
    2: "Forward head tilt! Pull chin back and align neck with shoulders.",
    3: "Shoulders misaligned! Level your shoulders evenly.",
}


class PostureClassifier:
    def __init__(self, model_path=None, calibration_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_path = model_path or os.path.join(base_dir, "backend", "models", "posture_model.joblib")
        self.metadata_path = os.path.join(base_dir, "backend", "models", "model_metadata.json")
        self.calibration_path = calibration_path or os.path.join(base_dir, "backend", "models", "calibration_profile.json")
        self.user_data_path = os.path.join(base_dir, "data", "user_posture_dataset.csv")

        self.feature_extractor = PostureFeatureExtractor()
        self.scaler = None
        self.model = None

        # Calibration state
        self.is_calibrated = False
        self.calibration_offset = np.zeros(38, dtype=np.float32)
        self.user_baseline = None
        self.calibration_samples = []
        self.is_calibrating = False
        self.calibration_target_frames = 60

        # Continuous user training metrics
        self.users_trained_count = 0
        self.total_training_samples = 0

        # Load or initialize model
        self._load_or_initialize_model()
        self._load_calibration()

    def _load_or_initialize_model(self):
        """Loads trained model or initializes learning pipeline."""
        if os.path.exists(self.model_path) and joblib:
            try:
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self.model = loaded["model"]
                    self.scaler = loaded.get("scaler")
                else:
                    self.model = loaded
                print(f"[MoveSure ML] Loaded trained posture model from {self.model_path}")
                self._load_metadata()
                return
            except Exception as e:
                print(f"[MoveSure ML] Error loading saved model: {e}. Reinitializing.")

        # If no saved model found, train baseline model from synthetic biomechanical data
        self._train_initial_model()

    def _train_initial_model(self):
        """Builds and trains baseline model using biomechanical distributions."""
        from backend.models.dataset_generator import generate_posture_samples

        print("[MoveSure ML] Training initial ergonomic baseline model...")
        X, y = generate_posture_samples(num_samples_per_class=400)

        if StandardScaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        if SGDClassifier:
            # SGDClassifier with log_loss supports incremental partial_fit per user
            self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=42)
            self.model.fit(X_scaled, y)
        elif RandomForestClassifier:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            self.model.fit(X_scaled, y)

        self.total_training_samples = len(X)
        self.save_model()
        print("[MoveSure ML] Baseline model trained successfully.")

    def save_model(self):
        """Persists trained model and training metadata to disk."""
        if joblib and self.model:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_path)

        metadata = {
            "total_samples": self.total_training_samples,
            "users_trained_count": self.users_trained_count,
            "classes": CLASS_LABELS,
            "last_updated": time.time(),
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                    self.total_training_samples = data.get("total_samples", 0)
                    self.users_trained_count = data.get("users_trained_count", 0)
            except Exception:
                pass

    # ==========================================
    # Calibration through Model Adaptation
    # ==========================================
    def start_calibration(self, target_frames=60):
        """Initiates personalized calibration for the active user."""
        self.is_calibrating = True
        self.calibration_samples = []
        self.calibration_target_frames = target_frames
        return {"status": "started", "target_frames": target_frames}

    def process_calibration_frame(self, landmarks):
        """
        Gathers user reference frames during calibration,
        computes personal neutral landmark distribution, and adapts model.
        """
        if not self.is_calibrating:
            return {"calibrated": self.is_calibrated}

        features = self.feature_extractor.extract_features(landmarks)
        if features is not None:
            self.calibration_samples.append(features)

        current_count = len(self.calibration_samples)
        if current_count >= self.calibration_target_frames:
            # Calibration complete: compute personal baseline
            samples = np.array(self.calibration_samples, dtype=np.float32)
            user_mean = np.mean(samples, axis=0)
            user_std = np.std(samples, axis=0) + 1e-4

            # Ergonomic Ideal Vector (Class 0 mean)
            ideal_neutral = np.zeros_like(user_mean)
            ideal_neutral[0] = 0.0    # Shoulder tilt
            ideal_neutral[1] = 0.0    # Shoulder delta
            ideal_neutral[4] = 12.0   # Neck angle
            ideal_neutral[7] = 5.0    # Torso angle
            ideal_neutral[8] = 1.25   # Head ratio
            ideal_neutral[9] = 0.65   # Ear drop
            ideal_neutral[10] = 0.04  # Forward head
            ideal_neutral[11] = 1.50  # Torso compression

            # Personalized calibration offset adapts features to user physiology & camera placement
            self.calibration_offset = user_mean - ideal_neutral
            self.user_baseline = {
                "mean": user_mean.tolist(),
                "std": user_std.tolist(),
                "calibrated_at": time.time(),
            }
            self.is_calibrated = True
            self.is_calibrating = False
            self._save_calibration()

            # Train/adapt model with user's calibrated neutral posture
            self.train_on_user_data(
                user_id="calibration_user",
                samples=self.calibration_samples,
                labels=[0] * len(self.calibration_samples)
            )

            return {
                "calibrated": True,
                "status": "completed",
                "frames_processed": current_count,
                "message": "Model calibrated successfully to your neutral posture."
            }

        return {
            "calibrated": False,
            "status": "in_progress",
            "progress_percent": int((current_count / self.calibration_target_frames) * 100),
            "frames_processed": current_count,
        }

    def _save_calibration(self):
        os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)
        data = {
            "is_calibrated": self.is_calibrated,
            "offset": self.calibration_offset.tolist(),
            "baseline": self.user_baseline,
        }
        with open(self.calibration_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_calibration(self):
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r") as f:
                    data = json.load(f)
                    self.is_calibrated = data.get("is_calibrated", False)
                    self.calibration_offset = np.array(data.get("offset", np.zeros(38)), dtype=np.float32)
                    self.user_baseline = data.get("baseline")
            except Exception:
                pass

    # ==========================================
    # Continuous User Model Training
    # ==========================================
    def train_on_user_data(self, user_id, samples, labels):
        """
        Updates the machine learning model using new user data.
        Ensures the model improves and adapts with each user who tests it.
        """
        if not samples or not labels:
            return False

        X_user = np.array(samples, dtype=np.float32)
        y_user = np.array(labels, dtype=np.int32)

        # Scale features
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X_user)
            except Exception:
                X_scaled = X_user
        else:
            X_scaled = X_user

        # Incremental model update
        if hasattr(self.model, "partial_fit"):
            try:
                self.model.partial_fit(X_scaled, y_user, classes=np.array([0, 1, 2, 3]))
            except Exception as e:
                print(f"[MoveSure ML] Error during partial_fit: {e}")
        elif hasattr(self.model, "fit"):
            # Refit / fine-tune model with combined data
            try:
                self.model.fit(X_scaled, y_user)
            except Exception as e:
                print(f"[MoveSure ML] Error during fit: {e}")

        # Update persistent stats
        self.total_training_samples += len(X_user)
        self.users_trained_count += 1
        self._append_to_user_dataset(user_id, X_user, y_user)
        self.save_model()
        print(f"[MoveSure ML] Model trained with {len(X_user)} samples from {user_id}. Total samples: {self.total_training_samples}")
        return True

    def _append_to_user_dataset(self, user_id, X, y):
        """Appends new user training samples to CSV for continuous record."""
        os.makedirs(os.path.dirname(self.user_data_path), exist_ok=True)
        file_exists = os.path.exists(self.user_data_path)
        try:
            with open(self.user_data_path, "a") as f:
                if not file_exists:
                    header = "user_id," + ",".join(PostureFeatureExtractor.FEATURE_NAMES) + ",label\n"
                    f.write(header)
                for i in range(len(X)):
                    feat_str = ",".join(f"{val:.4f}" for val in X[i])
                    f.write(f"{user_id},{feat_str},{y[i]}\n")
        except Exception as e:
            print(f"[MoveSure ML] Failed to write user dataset: {e}")

    # ==========================================
    # Real-Time Model Inference
    # ==========================================
    def predict(self, landmarks):
        """
        Classifies posture in real time using the trained ML model.
        Returns detailed ergonomic metrics, probabilities, and feedback.
        """
        raw_features = self.feature_extractor.extract_features(landmarks)
        if raw_features is None:
            return None

        angles = self.feature_extractor.get_angles_summary(raw_features)

        # Apply calibration offset learned from user neutral posture
        if self.is_calibrated:
            calibrated_features = raw_features - self.calibration_offset
        else:
            calibrated_features = raw_features

        features_input = calibrated_features.reshape(1, -1)

        # Predict using trained model
        class_id = 0
        probabilities = [1.0, 0.0, 0.0, 0.0]
        confidence = 0.95

        if self.model is not None:
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_input)
            else:
                features_scaled = features_input

            try:
                class_id = int(self.model.predict(features_scaled)[0])
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(features_scaled)[0]
                    probabilities = [float(p) for p in probs]
                    confidence = float(probabilities[class_id])
            except Exception as e:
                print(f"[MoveSure ML] Prediction error: {e}")
                # Biomechanical rule-based fallback if ML runtime exception occurs
                class_id = self._biomechanical_fallback(calibrated_features)

        # Posture health score (100 = perfect, lower when bad posture detected)
        is_good = (class_id == 0)
        posture_score = int(100 * probabilities[0]) if len(probabilities) > 0 else (95 if is_good else 35)

        return {
            "class_id": class_id,
            "class_name": CLASS_LABELS.get(class_id, "Unknown"),
            "is_good": is_good,
            "confidence": round(confidence, 3),
            "probabilities": {CLASS_LABELS[i]: round(probabilities[i], 3) for i in range(len(probabilities))},
            "posture_score": posture_score,
            "feedback": FEEDBACK_MESSAGES.get(class_id, ""),
            "angles": angles,
            "is_calibrated": self.is_calibrated,
            "raw_features": raw_features.tolist(),
        }

    def _biomechanical_fallback(self, f):
        """Ergonomic fallback if model pipeline is not yet fitted."""
        # f[0]: shoulder tilt, f[4]: neck angle, f[8]: head ratio, f[10]: fwd head
        if abs(f[0]) > 6.0:
            return 3  # Uneven shoulders
        if f[10] > 0.15 or f[4] > 26.0:
            return 2  # Forward head
        if f[8] < 0.95:
            return 1  # Slouching
        return 0
