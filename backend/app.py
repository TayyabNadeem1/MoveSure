"""
MoveSure Flask Web Backend
Serves the web dashboard, real-time video feed with ML posture classification overlay,
model-based calibration, and continuous user training endpoints.
"""

import os
import cv2
import threading
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from backend.utils.movement_detection import PostureDetector

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)
CORS(app)

# Initialize Posture Detector with ML classifier
camera_index = int(os.getenv("CAMERA_INDEX", 0))
detector = PostureDetector(camera_index=camera_index)

cap = None
camera_lock = threading.Lock()
is_monitoring = False


def get_camera():
    global cap
    with camera_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    return cap


def generate_frames():
    global cap, is_monitoring
    is_monitoring = True
    camera = get_camera()

    while is_monitoring:
        with camera_lock:
            if not camera.isOpened():
                break
            ret, frame = camera.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        processed_frame, _ = detector.process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/video_feed")
def video_feed():
    camera = get_camera()
    if not camera.isOpened():
        return jsonify({"error": "Could not access the camera"}), 500
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/calibrate", methods=["POST"])
def calibrate_posture():
    """Triggers ML model-based neutral posture calibration for user."""
    target_frames = request.json.get("frames", 60) if request.is_json else 60
    result = detector.classifier.start_calibration(target_frames=target_frames)
    return jsonify({
        "status": "success",
        "message": f"Calibration started. Please sit upright for {target_frames} frames.",
        "data": result
    })


@app.route("/train-user", methods=["POST"])
def train_user():
    """
    Receives user-labeled posture samples and immediately retrains/updates the ML model.
    Enables continuous learning with each new user test.
    """
    data = request.get_json() or {}
    user_id = data.get("user_id", "user_web")
    samples = data.get("samples", [])
    labels = data.get("labels", [])

    if not samples or not labels:
        return jsonify({"status": "error", "message": "Samples and labels required."}), 400

    success = detector.classifier.train_on_user_data(user_id, samples, labels)
    return jsonify({
        "status": "success" if success else "error",
        "message": f"Model updated with {len(samples)} user samples.",
        "total_training_samples": detector.classifier.total_training_samples,
        "users_trained_count": detector.classifier.users_trained_count
    })


@app.route("/posture-status", methods=["GET"])
def posture_status():
    """Returns real-time posture stats, angles, and score for charts and UI."""
    stats = detector.get_session_stats()
    stats["model_metadata"] = {
        "users_trained_count": detector.classifier.users_trained_count,
        "total_training_samples": detector.classifier.total_training_samples,
        "is_calibrated": detector.classifier.is_calibrated,
    }
    return jsonify(stats)


@app.route("/stop-monitoring", methods=["POST", "GET"])
def stop_monitoring():
    global cap, is_monitoring
    is_monitoring = False
    with camera_lock:
        if cap and cap.isOpened():
            cap.release()
            cap = None
    return jsonify({"status": "success", "message": "Posture monitoring stopped."})


@app.teardown_appcontext
def cleanup(exception=None):
    global cap
    with camera_lock:
        if cap and cap.isOpened():
            cap.release()
            cap = None


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
