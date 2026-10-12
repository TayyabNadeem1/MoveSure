"""
MoveSure Desktop Application Entry Point
Starts the real-time AI posture detector with trained machine learning model,
adaptive user calibration, and live HUD.

Usage:
    python main.py [--camera 0] [--no-audio]
"""

import argparse
from backend.utils.movement_detection import PostureDetector


def main():
    parser = argparse.ArgumentParser(description="MoveSure AI Posture Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--no-audio", action="store_true", help="Disable speech feedback alerts")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Audio alert cooldown in seconds")
    args = parser.parse_args()

    detector = PostureDetector(
        camera_index=args.camera,
        enable_audio=not args.no_audio,
        audio_cooldown=args.cooldown
    )
    detector.run_desktop_stream()


if __name__ == "__main__":
    main()
