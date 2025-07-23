# 🧠 MoveSure: AI-Based Posture & Movement Monitoring System

<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/MediaPipe-FE6B00?style=for-the-badge&logo=google&logoColor=white" />
<img src="https://img.shields.io/badge/Pygame-000000?style=for-the-badge&logo=pygame&logoColor=white" />
<img src="https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=node.js&logoColor=white" />

</div>

**MoveSure** is an AI-powered posture monitoring and correction system designed to help individuals—such as workers, students, or anyone sitting for long periods—maintain healthy posture habits. Using real-time computer vision, MoveSure detects poor posture and slouching through a webcam feed and delivers immediate feedback to the user.

---

## ⚙️ Features

- ✅ **Real-Time Posture Detection**  
  Uses computer vision (MediaPipe + OpenCV) to detect human pose landmarks and monitor posture in real time.

- 🪑 **Slouching Detection**  
  Measures the distance between key landmarks (mouth and shoulders) to detect slouching. Alerts users when the posture crosses a defined threshold.

- 📐 **Neck, Shoulder, and Spine Angle Monitoring**  
  Continuously tracks posture angles to spot unhealthy patterns or strains.

- 🔴 **Feedback & Alerts**  
  - Red screen + “Poor Posture” message for incorrect posture  
  - Green screen + “Good Posture” when posture is healthy  
  - Optional audio alerts for extra awareness (via Pygame)

- 🎯 **Posture Calibration**  
  Automatically calibrates neutral sitting posture by having the user sit still for a few seconds at startup.

- 📊 **Posture History (Optional)**  
  Logs posture behavior over time for tracking improvements or identifying trends.

---

## 📸 UI/UX

<div align="center">

<table>
<tr>
<td align="center"><strong>Login Page</strong><br><img src="https://github.com/user-attachments/assets/9fc4a9bd-327c-4a86-a6bb-63636d3e2e57" width="400"/></td>
<td align="center"><strong>Sign Up Page</strong><br><img src="https://github.com/user-attachments/assets/c5473143-4b08-4966-9a64-41b3c0765f81" width="400"/></td>
</tr>
<tr>
<td align="center"><strong>Home Page</strong><br><img src="https://github.com/user-attachments/assets/d21c2c67-d69c-4062-976e-31409445db0c" width="400"/></td>
<td align="center"><strong>Profile Page</strong><br><img src="https://github.com/user-attachments/assets/b365dde1-2719-49ad-85ed-021da9955c1c" width="400"/></td>
</tr>
<tr>
<td colspan="2" align="center"><strong>Report Page</strong><br><img src="https://github.com/user-attachments/assets/52dfb575-62e2-437f-a2b8-5dcc3e26f867" width="600"/></td>
</tr>
</table>

</div>

---

## 🛠 Installation

### 📋 Requirements

- Python 3.x  
- OpenCV  
- MediaPipe  
- Numpy  
- Pygame  
- Node.js (for backend/server integration)

---

## 🛠 Installation

### 📋 Requirements

- Python 3.x  
- OpenCV  
- MediaPipe  
- Numpy  
- Pygame  
- Node.js (for backend/server integration)

---

### 💻 Installation Steps:
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
