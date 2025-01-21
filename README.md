# MoveSure: AI-Based Person Movement and Posture Monitoring

MoveSure is an AI-powered posture correction system designed to monitor and improve the posture of individuals, such as workers, students, or anyone sitting for prolonged periods. The system uses computer vision to detect body movements and postures via camera feeds and provides real-time feedback to prevent back pain and promote healthy sitting habits.

## Features

- **Real-Time Posture Detection**: Using AI and computer vision techniques, the system analyzes the posture of the user and detects when it deviates from the correct posture.
  
- **Slouching Detection**: The system identifies slouching by measuring the distance between the user's mouth and shoulder midpoints. When the user slouches beyond a predefined threshold, it alerts them to correct their posture.

- **Neck, Shoulder, and Spine Angle Monitoring**: The system monitors the angles of key body parts, including the neck, shoulders, and spine, to detect misalignments or strains.

- **Feedback and Alerts**: When poor posture or slouching is detected, the system provides real-time visual feedback on the screen, turning it red and displaying a "Poor Posture" message. It can also trigger audio alerts for immediate attention.

- **Posture Calibration**: The system allows the user to calibrate their posture by staying in a neutral position for a few seconds at the start. This calibration helps the system identify a baseline for good posture.

- **Posture History and Reporting**: MoveSure can log posture data over time, helping users track their progress and identify patterns that lead to poor posture.

## Installation

### Requirements:
- Python 3.x
- OpenCV
- MediaPipe
- Numpy
- Pygame
- Node.js

### Installation Steps:
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/movesure.git
    ```

2. Navigate to the project directory:
    ```bash
    cd movesure
    ```

3. Install the necessary Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. Install Node.js dependencies:
    ```bash
    cd node_app
    npm install
    ```

5. Ensure that your system's camera or a webcam is accessible to the program.

## Usage

### Running the Python Application:

1. **Open a terminal** window and run the Python application:
    ```bash
    python main.py
    ```

2. The program will initialize the camera and start detecting the user's posture in real time.

3. If the user maintains poor posture or slouches, the screen will turn **red** and display a "Poor Posture" message. If the posture is corrected, it will display **green** and show "Good Posture."

4. The system also gives audio alerts when needed to remind users to correct their posture.

### Running the Node.js Application:

1. **Open another terminal** window (or a new tab) and navigate to the `node_app` directory:
    ```bash
    cd movesure/node_app
    ```

2. Run the Node.js application with:
    ```bash
    node app.js
    ```

This will start the Node.js server that integrates with the Python posture monitoring system. 

## Customization

- **Sensitivity**: You can adjust the sensitivity of slouching detection by modifying the `slouching_threshold_factor` value in the code.
- **Audio Alerts**: You can change the audio file for alerts by updating the `sound_file` variable in the code.
- **Node.js Server**: Modify the `app.js` file to fit your specific use case if you need to adjust the server-side functionality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to [MediaPipe](https://google.github.io/mediapipe/) for the pose detection models.
- The project uses [OpenCV](https://opencv.org/) for computer vision tasks and [Pygame](https://www.pygame.org/) for audio alerts.

---

### Key Points:
- **Python Application**: Run in one terminal window using `python main.py`.
- **Node.js Application**: Run in another terminal window using `node app.js`.

This setup will allow you to run both applications simultaneously.
