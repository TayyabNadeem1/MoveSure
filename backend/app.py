import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
import os
import threading
from gtts import gTTS
from playsound import playsound

# Get the absolute path of the project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, '..', 'frontend', 'templates'),
    static_folder=os.path.join(BASE_DIR, '..', 'frontend', 'static')
)
print("Template Folder:", os.path.join(BASE_DIR, '..', 'frontend', 'templates'))
print("Static Folder:", os.path.join(BASE_DIR, '..', 'frontend', 'static'))

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing_utils = mp.solutions.drawing_utils

# Camera initialization
cap = None
camera_lock = threading.Lock()

# Helper to initialize the camera
def initialize_camera():
    global cap
    with camera_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Default camera
            if not cap.isOpened():
                print("Error: Could not access the camera.")
                return False
    return True

# Function to provide speech feedback
def speech_feedback(message):
    try:
        print(f"Alert: {message}")
        tts = gTTS(text=message, lang="en", slow=False)
        sound_path = os.path.join('frontend', 'sounds', 'feedback.mp3')
        tts.save(sound_path)
        playsound(sound_path)
    except Exception as e:
        print(f"Speech feedback error: {e}")

# Monitor posture in real-time
def check_posture(landmarks):
    try:
        # Example: Check shoulder alignment (you can refine this with more checks)
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Check if shoulders are misaligned (example condition)
        if abs(left_shoulder.y - right_shoulder.y) > 0.05:  # Arbitrary threshold for misalignment
            speech_feedback("Please align your shoulders!")
    except Exception as e:
        print(f"Error in posture check: {e}")

# Video frame generator
def generate_frames():
    global cap
    while True:
        with camera_lock:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw landmarks on the frame
            if results.pose_landmarks:
                drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                check_posture(results.pose_landmarks)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()

            # Yield the frame to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not initialize_camera():
        return "Error: Could not access the camera.", 500
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Cleanup resources on app exit
@app.teardown_appcontext
def cleanup(exception=None):
    global cap
    if cap and cap.isOpened():
        cap.release()

@app.route('/stop-monitoring', methods=['GET'])
def stop_monitoring():
    try:
        # Logic to stop the camera and monitoring process
        global cap
        if cap and cap.isOpened():
            cap.release()
        return {"message": "Posture monitoring has been stopped."}
    except Exception as e:
        return {"error": str(e)}, 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
