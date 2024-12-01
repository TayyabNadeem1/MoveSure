from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the camera
cap = cv2.VideoCapture(0)  # Open webcam (change to 1 if using external camera)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Convert frame to JPEG for web display
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        
        # Yield the frame in the correct format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Serve the video feed with posture landmarks
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start-monitoring', methods=['GET'])
def start_monitoring():
    try:
        return jsonify({"message": "Posture monitoring is now running."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
