"""
MoveSure Feature Extractor
Extracts normalized, scale-invariant, and camera-distance invariant
biomechanical features from MediaPipe Pose landmarks for ML classification.
"""

import math
import numpy as np


class PostureFeatureExtractor:
    FEATURE_NAMES = [
        "shoulder_tilt_deg",
        "shoulder_y_delta_norm",
        "neck_angle_left_deg",
        "neck_angle_right_deg",
        "neck_angle_avg_deg",
        "torso_angle_left_deg",
        "torso_angle_right_deg",
        "torso_angle_avg_deg",
        "head_to_shoulder_ratio",
        "ear_to_shoulder_drop_norm",
        "forward_head_offset_norm",
        "torso_compression_ratio",
        # Normalized (x, y) coordinates relative to shoulder center and span
        "nose_rel_x", "nose_rel_y",
        "left_eye_rel_x", "left_eye_rel_y",
        "right_eye_rel_x", "right_eye_rel_y",
        "left_ear_rel_x", "left_ear_rel_y",
        "right_ear_rel_x", "right_ear_rel_y",
        "left_shoulder_rel_x", "left_shoulder_rel_y",
        "right_shoulder_rel_x", "right_shoulder_rel_y",
        "left_elbow_rel_x", "left_elbow_rel_y",
        "right_elbow_rel_x", "right_elbow_rel_y",
        "left_wrist_rel_x", "left_wrist_rel_y",
        "right_wrist_rel_x", "right_wrist_rel_y",
        "left_hip_rel_x", "left_hip_rel_y",
        "right_hip_rel_x", "right_hip_rel_y",
    ]

    @staticmethod
    def calculate_angle(p1, p2, p3=None):
        """
        Calculate angle in degrees.
        If p3 is None, calculates angle of vector (p1 -> p2) against vertical axis.
        """
        if p3 is None:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Angle with respect to downward vertical (0 deg is straight down)
            angle = math.degrees(math.atan2(dx, dy))
            return abs(angle)

        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 * mag2 == 0:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    @classmethod
    def extract_features(cls, landmarks):
        """
        Takes MediaPipe pose landmarks (or a dict/list of landmarks with x, y, z)
        and extracts a normalized feature vector.
        """
        if landmarks is None:
            return None

        # Support both mediapipe LandmarkList and list/dict representations
        def get_pt(idx):
            if hasattr(landmarks, "landmark"):
                lm = landmarks.landmark[idx]
                return (lm.x, lm.y, getattr(lm, "z", 0.0))
            elif isinstance(landmarks, (list, tuple)):
                lm = landmarks[idx]
                if isinstance(lm, dict):
                    return (lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0))
                return (lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0)
            return (0.0, 0.0, 0.0)

        # Key Landmarks from MediaPipe Pose
        # 0: Nose, 2: Left Eye, 5: Right Eye, 7: Left Ear, 8: Right Ear
        # 11: Left Shoulder, 12: Right Shoulder, 13: Left Elbow, 14: Right Elbow
        # 15: Left Wrist, 16: Right Wrist, 23: Left Hip, 24: Right Hip
        nose = get_pt(0)
        l_eye = get_pt(2)
        r_eye = get_pt(5)
        l_ear = get_pt(7)
        r_ear = get_pt(8)
        l_sh = get_pt(11)
        r_sh = get_pt(12)
        l_el = get_pt(13)
        r_el = get_pt(14)
        l_wr = get_pt(15)
        r_wr = get_pt(16)
        l_hip = get_pt(23)
        r_hip = get_pt(24)

        # Baseline shoulder span for distance-invariant normalization
        sh_span = math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
        if sh_span < 1e-4:
            sh_span = 0.1  # Fallback epsilon to prevent division by zero

        # Midpoints
        sh_center = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        hip_center = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        ear_center = ((l_ear[0] + r_ear[0]) / 2.0, (l_ear[1] + r_ear[1]) / 2.0)

        # 1. Shoulder tilt angle (horizontal = 0 deg)
        sh_dx = r_sh[0] - l_sh[0]
        sh_dy = r_sh[1] - l_sh[1]
        shoulder_tilt_deg = math.degrees(math.atan2(sh_dy, sh_dx))

        # 2. Shoulder y delta normalized by shoulder span
        shoulder_y_delta_norm = (l_sh[1] - r_sh[1]) / sh_span

        # 3. Neck angles (ear -> shoulder -> vertical)
        neck_angle_left_deg = cls.calculate_angle(l_ear, l_sh)
        neck_angle_right_deg = cls.calculate_angle(r_ear, r_sh)
        neck_angle_avg_deg = (neck_angle_left_deg + neck_angle_right_deg) / 2.0

        # 4. Torso inclination angles (shoulder -> hip -> vertical)
        torso_angle_left_deg = cls.calculate_angle(l_sh, l_hip)
        torso_angle_right_deg = cls.calculate_angle(r_sh, r_hip)
        torso_angle_avg_deg = (torso_angle_left_deg + torso_angle_right_deg) / 2.0

        # 5. Head-to-shoulder height ratio
        head_sh_dist = abs(sh_center[1] - nose[1])
        head_to_shoulder_ratio = head_sh_dist / sh_span

        # 6. Ear-to-shoulder vertical drop
        ear_sh_drop = ((l_sh[1] - l_ear[1]) + (r_sh[1] - r_ear[1])) / (2.0 * sh_span)

        # 7. Forward head offset (horizontal difference between ear and shoulder)
        forward_head_offset = abs(ear_center[0] - sh_center[0]) / sh_span

        # 8. Torso compression ratio (shoulder center to hip center distance)
        torso_len = math.hypot(hip_center[0] - sh_center[0], hip_center[1] - sh_center[1])
        torso_compression_ratio = torso_len / sh_span

        # 9. Normalized keypoint coordinates centered at shoulder midpoint
        def norm_pt(pt):
            return [
                (pt[0] - sh_center[0]) / sh_span,
                (pt[1] - sh_center[1]) / sh_span,
            ]

        keypoint_features = []
        for pt in [nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip]:
            keypoint_features.extend(norm_pt(pt))

        feature_vector = [
            shoulder_tilt_deg,
            shoulder_y_delta_norm,
            neck_angle_left_deg,
            neck_angle_right_deg,
            neck_angle_avg_deg,
            torso_angle_left_deg,
            torso_angle_right_deg,
            torso_angle_avg_deg,
            head_to_shoulder_ratio,
            ear_sh_drop,
            forward_head_offset,
            torso_compression_ratio,
            *keypoint_features,
        ]

        return np.array(feature_vector, dtype=np.float32)

    @classmethod
    def get_angles_summary(cls, feature_vector):
        """
        Extract human-readable angles from feature vector for display / HUD.
        """
        if feature_vector is None:
            return {}
        return {
            "shoulder_tilt": round(float(feature_vector[0]), 1),
            "neck_angle": round(float(feature_vector[4]), 1),
            "torso_angle": round(float(feature_vector[7]), 1),
            "head_ratio": round(float(feature_vector[8]), 2),
        }
