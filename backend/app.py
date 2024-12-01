from flask import Flask, jsonify
import cv2
import mediapipe as mp
from flask import render_template
app = Flask(__name__)

@app.route('/start-monitoring', methods=['GET'])
def start_monitoring():
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        cap = cv2.VideoCapture(0)  # Open webcam

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw landmarks
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow('MoveSure - Posture Monitoring', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()
        return jsonify({"message": "Posture monitoring ended."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
