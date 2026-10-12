"""
MoveSure Biomechanical Posture Dataset Generator
Generates realistic baseline ergonomic training data across diverse body types,
camera distances, and posture states for machine learning training.
"""

import numpy as np
import pandas as pd
from backend.models.feature_extractor import PostureFeatureExtractor

CLASS_NAMES = ["Good Posture", "Slouching", "Forward Head", "Uneven Shoulders"]
CLASS_LABELS = {
    0: "Good Posture",
    1: "Slouching",
    2: "Forward Head",
    3: "Uneven Shoulders",
}


def generate_posture_samples(num_samples_per_class=600, random_seed=42):
    """
    Synthesizes realistic MediaPipe-like normalized 38-dimensional feature vectors
    grounded in clinical ergonomic ergonomics:
    - Class 0 (Good Posture): Neck 10-18°, Shoulders level (tilt < 2°), Upright torso (0-8°), High head ratio
    - Class 1 (Slouching): Collapsed torso (compression ratio decreases), low head ratio, neck drop
    - Class 2 (Forward Head): Forward head offset increases, neck angle exceeds 28-45°
    - Class 3 (Uneven Shoulders): Shoulder tilt angle 6-20°, high normalized y-delta
    """
    rng = np.random.RandomState(random_seed)
    features_list = []
    labels = []

    for class_id in range(4):
        for _ in range(num_samples_per_class):
            # Scale variations simulating different distances from camera
            scale = rng.uniform(0.85, 1.15)
            noise = rng.normal(0, 0.02, size=38)

            if class_id == 0:  # Good Posture
                shoulder_tilt = rng.normal(0.0, 1.5)
                shoulder_y_delta = rng.normal(0.0, 0.015)
                neck_left = rng.normal(12.0, 2.5)
                neck_right = rng.normal(12.0, 2.5)
                neck_avg = (neck_left + neck_right) / 2.0
                torso_left = rng.normal(5.0, 2.0)
                torso_right = rng.normal(5.0, 2.0)
                torso_avg = (torso_left + torso_right) / 2.0
                head_ratio = rng.normal(1.25, 0.08)
                ear_drop = rng.normal(0.65, 0.05)
                fwd_head = rng.normal(0.04, 0.02)
                torso_comp = rng.normal(1.50, 0.08)

            elif class_id == 1:  # Slouching
                shoulder_tilt = rng.normal(1.0, 3.0)
                shoulder_y_delta = rng.normal(0.02, 0.03)
                neck_left = rng.normal(24.0, 4.0)
                neck_right = rng.normal(24.0, 4.0)
                neck_avg = (neck_left + neck_right) / 2.0
                torso_left = rng.normal(18.0, 4.0)
                torso_right = rng.normal(18.0, 4.0)
                torso_avg = (torso_left + torso_right) / 2.0
                head_ratio = rng.normal(0.85, 0.07)  # Head visibly sunk into shoulders
                ear_drop = rng.normal(0.40, 0.05)
                fwd_head = rng.normal(0.12, 0.04)
                torso_comp = rng.normal(1.15, 0.07)  # Compressed spine

            elif class_id == 2:  # Forward Head
                shoulder_tilt = rng.normal(0.0, 2.0)
                shoulder_y_delta = rng.normal(0.0, 0.02)
                neck_left = rng.normal(32.0, 5.0)  # High anterior cervical tilt
                neck_right = rng.normal(32.0, 5.0)
                neck_avg = (neck_left + neck_right) / 2.0
                torso_left = rng.normal(7.0, 3.0)
                torso_right = rng.normal(7.0, 3.0)
                torso_avg = (torso_left + torso_right) / 2.0
                head_ratio = rng.normal(1.05, 0.08)
                ear_drop = rng.normal(0.50, 0.06)
                fwd_head = rng.normal(0.22, 0.05)  # Significant forward protrusion
                torso_comp = rng.normal(1.42, 0.07)

            else:  # Uneven Shoulders
                tilt_sign = rng.choice([-1.0, 1.0])
                shoulder_tilt = rng.normal(tilt_sign * 11.0, 2.5)  # Severe tilt
                shoulder_y_delta = rng.normal(tilt_sign * 0.12, 0.03)
                neck_left = rng.normal(18.0, 3.5)
                neck_right = rng.normal(18.0, 3.5)
                neck_avg = (neck_left + neck_right) / 2.0
                torso_left = rng.normal(9.0, 3.0)
                torso_right = rng.normal(9.0, 3.0)
                torso_avg = (torso_left + torso_right) / 2.0
                head_ratio = rng.normal(1.18, 0.08)
                ear_drop = rng.normal(0.60, 0.06)
                fwd_head = rng.normal(0.06, 0.03)
                torso_comp = rng.normal(1.44, 0.08)

            # Simulated normalized keypoints centered around standard pose
            kp = [
                0.0, -head_ratio,                      # nose
                -0.12, -head_ratio + 0.05,             # l_eye
                0.12, -head_ratio + 0.05,              # r_eye
                -0.35, -head_ratio + 0.15,             # l_ear
                0.35, -head_ratio + 0.15,              # r_ear
                -0.5, shoulder_y_delta / 2.0,          # l_sh
                0.5, -shoulder_y_delta / 2.0,          # r_sh
                -0.65, 0.55,                           # l_el
                0.65, 0.55,                            # r_el
                -0.60, 1.05,                           # l_wr
                0.60, 1.05,                            # r_wr
                -0.35, torso_comp,                     # l_hip
                0.35, torso_comp,                      # r_hip
            ]

            sample = [
                shoulder_tilt,
                shoulder_y_delta,
                neck_left,
                neck_right,
                neck_avg,
                torso_left,
                torso_right,
                torso_avg,
                head_ratio,
                ear_drop,
                fwd_head,
                torso_comp,
                *kp,
            ]
            sample = np.array(sample, dtype=np.float32) + noise
            features_list.append(sample)
            labels.append(class_id)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def save_dataset_csv(filepath="data/baseline_posture_dataset.csv"):
    X, y = generate_posture_samples()
    df = pd.DataFrame(X, columns=PostureFeatureExtractor.FEATURE_NAMES)
    df["label"] = y
    df["class_name"] = [CLASS_LABELS[i] for i in y]
    df.to_csv(filepath, index=False)
    print(f"Dataset generated with {len(df)} samples at {filepath}")
    return filepath


if __name__ == "__main__":
    save_dataset_csv()
