"""
MoveSure Interactive Posture Data Collector & Model Trainer
Enables new users to record their postures and immediately train the ML model.
"""

import cv2
import mediapipe as mp
import time
import argparse
from backend.models.posture_classifier import PostureClassifier, CLASS_LABELS
from backend.models.feature_extractor import PostureFeatureExtractor


def record_and_train_user(user_id="user_test", camera_id=0):
    print("=" * 60)
    print(f" MoveSure User Posture Training & Data Collector ({user_id})")
    print("=" * 60)
    print("Instructions:")
    print("  Press '0': Record Good Posture samples")
    print("  Press '1': Record Slouching samples")
    print("  Press '2': Record Forward Head samples")
    print("  Press '3': Record Uneven Shoulders samples")
    print("  Press 'C': Calibrate Model Neutral Baseline (3 seconds)")
    print("  Press 'T': Train / Update Model with recorded samples")
    print("  Press 'Q': Quit")
    print("=" * 60)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_utils = mp.solutions.drawing_utils

    classifier = PostureClassifier()
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    collected_samples = []
    collected_labels = []
    active_recording_class = None
    frames_for_current_class = 0
    target_frames = 60

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks

            # Handle automatic recording bursts
            if active_recording_class is not None:
                features = classifier.feature_extractor.extract_features(landmarks)
                if features is not None:
                    collected_samples.append(features)
                    collected_labels.append(active_recording_class)
                    frames_for_current_class += 1

                progress = int((frames_for_current_class / target_frames) * 100)
                cv2.putText(frame, f"RECORDING {CLASS_LABELS[active_recording_class]}: {progress}%",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if frames_for_current_class >= target_frames:
                    print(f"[+] Finished recording {target_frames} frames for {CLASS_LABELS[active_recording_class]}")
                    active_recording_class = None
                    frames_for_current_class = 0
            else:
                # Normal live prediction
                pred = classifier.predict(landmarks)
                if pred:
                    color = (0, 255, 0) if pred["is_good"] else (0, 0, 255)
                    cv2.putText(frame, f"Model: {pred['class_name']} ({int(pred['confidence']*100)}%)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, f"Score: {pred['posture_score']}%",
                                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Bottom HUD
        cv2.putText(frame, f"Samples recorded: {len(collected_samples)} | Press 0,1,2,3 to record | 'T' to train",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("MoveSure - User Training Studio", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            active_recording_class = int(chr(key))
            frames_for_current_class = 0
            print(f"[*] Recording 60 frames for: {CLASS_LABELS[active_recording_class]}...")
        elif key == ord('c'):
            print("[*] Calibrating neutral posture...")
            classifier.start_calibration(target_frames=60)
        elif key == ord('t'):
            if collected_samples:
                print(f"[*] Training model with {len(collected_samples)} user samples...")
                classifier.train_on_user_data(user_id, collected_samples, collected_labels)
                print("[+] Model updated and saved! It will now recognize your personalized postures.")
                collected_samples = []
                collected_labels = []
            else:
                print("[!] No samples recorded yet. Press 0, 1, 2, or 3 first.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoveSure User Training Studio")
    parser.add_argument("--user", type=str, default="user_session_1", help="User identifier")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    record_and_train_user(user_id=args.user, camera_id=args.camera)
