"""
MoveSure Movement & Posture Detection Engine
Integrates MediaPipe Pose estimation with trained Machine Learning classification.
Provides real-time visual overlays, model-based calibration, and non-blocking speech alerts.
"""

import cv2
import mediapipe as mp
import time
import os
import threading
from backend.models.posture_classifier import PostureClassifier

try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
except Exception:
    pass


class PostureDetector:
    def __init__(self, camera_index=0, enable_audio=True, audio_cooldown=5.0):
        self.camera_index = camera_index
        self.enable_audio = enable_audio
        self.audio_cooldown = audio_cooldown
        self.last_audio_time = 0.0

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

        # Machine Learning Classifier & Calibration Engine
        self.classifier = PostureClassifier()

        # Session tracking statistics
        self.total_frames = 0
        self.good_frames = 0
        self.bad_frames = 0
        self.current_state = None

    def trigger_speech_alert(self, text):
        """Asynchronously triggers voice alert without blocking video pipeline."""
        if not self.enable_audio:
            return

        now = time.time()
        if now - self.last_audio_time < self.audio_cooldown:
            return

        self.last_audio_time = now

        def _play():
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                sound_dir = os.path.join(base_dir, "frontend", "sounds")
                os.makedirs(sound_dir, exist_ok=True)
                sound_path = os.path.join(sound_dir, "alert.mp3")

                # Generate TTS alert file
                tts = gTTS(text=text, lang="en", slow=False)
                tts.save(sound_path)

                if pygame and pygame.mixer.get_init():
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
            except Exception as e:
                print(f"[MoveSure Audio] Alert error: {e}")

        threading.Thread(target=_play, daemon=True).start()

    def process_frame(self, frame):
        """
        Processes a single BGR video frame:
        - Detects pose landmarks via MediaPipe.
        - Predicts posture class and confidence using trained ML model.
        - Updates continuous session metrics.
        - Draws HUD overlay and landmark skeleton.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        prediction = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks

            # Check if active calibration is ongoing
            if self.classifier.is_calibrating:
                calib_res = self.classifier.process_calibration_frame(landmarks)
                progress = calib_res.get("progress_percent", 0)
                # Calibration HUD overlay
                cv2.rectangle(frame, (10, 10), (w - 10, 90), (40, 40, 40), -1)
                cv2.putText(frame, f"CALIBRATING NEUTRAL POSTURE: {progress}%",
                            (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(frame, "Please sit in your natural upright posture...",
                            (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            else:
                # Classify posture using ML model
                prediction = self.classifier.predict(landmarks)
                self.current_state = prediction

                if prediction:
                    self.total_frames += 1
                    if prediction["is_good"]:
                        self.good_frames += 1
                    else:
                        self.bad_frames += 1
                        if prediction["feedback"]:
                            self.trigger_speech_alert(prediction["feedback"])

                    # Draw ergonomic HUD
                    self._draw_hud(frame, prediction)

            # Draw pose skeleton
            self.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            cv2.putText(frame, "No person detected in frame", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame, prediction

    def _draw_hud(self, frame, pred):
        """Draws clean ergonomic visual feedback HUD on frame."""
        h, w, _ = frame.shape
        is_good = pred["is_good"]
        primary_color = (46, 204, 113) if is_good else (52, 73, 235)  # Green vs Red (BGR)

        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Status text & confidence
        status_text = f"{pred['class_name'].upper()} ({int(pred['confidence'] * 100)}%)"
        cv2.putText(frame, status_text, (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, primary_color, 2)

        # Feedback instruction
        cv2.putText(frame, pred["feedback"], (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

        # Score badge (top right)
        score_text = f"Score: {pred['posture_score']}%"
        cv2.putText(frame, score_text, (w - 180, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, primary_color, 2)

        calib_tag = "ML Calibrated" if pred["is_calibrated"] else "Default Model (Press C)"
        cv2.putText(frame, calib_tag, (w - 230, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Bottom metrics bar
        angles = pred.get("angles", {})
        bottom_text = (
            f"Neck: {angles.get('neck_angle', 0)} deg | "
            f"Shoulder Tilt: {angles.get('shoulder_tilt', 0)} deg | "
            f"Torso: {angles.get('torso_angle', 0)} deg"
        )
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, bottom_text, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    def get_session_stats(self):
        """Returns session summary percentages for dashboard."""
        if self.total_frames == 0:
            return {"good_percent": 100, "bad_percent": 0, "total_frames": 0}
        good_pct = round((self.good_frames / self.total_frames) * 100, 1)
        bad_pct = round((self.bad_frames / self.total_frames) * 100, 1)
        return {
            "good_percent": good_pct,
            "bad_percent": bad_pct,
            "total_frames": self.total_frames,
            "current_state": self.current_state,
        }

    def run_desktop_stream(self):
        """Runs interactive desktop window with keyboard controls."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[MoveSure] Could not open camera {self.camera_index}")
            return

        print("\n" + "=" * 50)
        print(" MoveSure Posture Monitoring Active")
        print("   - Press 'C': Calibrate model to your sitting posture")
        print("   - Press 'Q': Quit")
        print("=" * 50 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame, _ = self.process_frame(frame)

            cv2.imshow("MoveSure Posture Monitor (ML Powered)", processed_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                print("[MoveSure] Triggering 3-second neutral posture model calibration...")
                self.classifier.start_calibration(target_frames=60)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PostureDetector()
    detector.run_desktop_stream()
