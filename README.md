# 🧠 MoveSure: Machine Learning Posture & Movement Monitoring System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose_Estimation-FE6B00?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_Application-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Node.js](https://img.shields.io/badge/Node.js-Express_API-339933?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org/)

<p align="center">
  <strong>An AI-powered ergonomic posture monitoring system that replaces static hardcoded thresholds with trained machine learning models, personalized adaptive calibration, and continuous learning from each user session.</strong>
</p>

</div>

---

## 📌 Overview

**MoveSure** is designed to prevent musculoskeletal fatigue and repetitive strain injuries for desk workers and students. While early computer vision projects rely on brittle, hardcoded distance rules (such as `delta > 0.05`), MoveSure introduces a **model-centric approach**:

1. **Trained ML Models**: Posture is classified by machine learning models trained on scale-invariant biomechanical features.
2. **Model-Based Calibration**: Rather than guessing fixed thresholds, the model observes your natural posture for 3 seconds and personalizes its decision boundary to your unique physiology and camera angle.
3. **Continuous Learning**: Every user session actively enriches the dataset. The model learns incrementally from new user tests, continuously expanding its accuracy across diverse body types.

---

## 📸 User Interface

<div align="center">

<table>
<tr>
<td align="center"><strong>Sign In Page</strong><br><img src="https://github.com/user-attachments/assets/9fc4a9bd-327c-4a86-a6bb-63636d3e2e57" width="400"/></td>
<td align="center"><strong>Registration Page</strong><br><img src="https://github.com/user-attachments/assets/c5473143-4b08-4966-9a64-41b3c0765f81" width="400"/></td>
</tr>
<tr>
<td align="center"><strong>Home Analytics Dashboard</strong><br><img src="https://github.com/user-attachments/assets/d21c2c67-d69c-4062-976e-31409445db0c" width="400"/></td>
<td align="center"><strong>User Profile</strong><br><img src="https://github.com/user-attachments/assets/b365dde1-2719-49ad-85ed-021da9955c1c" width="400"/></td>
</tr>
<tr>
<td colspan="2" align="center"><strong>Weekly Posture Report</strong><br><img src="https://github.com/user-attachments/assets/52dfb575-62e2-437f-a2b8-5dcc3e26f867" width="600"/></td>
</tr>
</table>

</div>

---

## ⚙️ Core Architecture & Innovation

```
                      ┌──────────────────────────────────────┐
                      │          Video Camera Feed           │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    MediaPipe 33-Landmark Pose Extr.  │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │ 38-D Biomechanical Feature Extractor │
                      │  - Shoulder Tilt & Delta (Norm.)     │
                      │  - Neck Flexion Angles (L / R / Avg) │
                      │  - Torso Compression & Inclination   │
                      │  - Head-to-Shoulder Ratio            │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴────────────────────────┐
                 ▼                                                ▼
     [Model-Based Calibration]                       [Trained ML Classifier]
   - Captures User Baseline (μ, σ)                  - Online SGD / Random Forest
   - Adjusts Personal Feature Offset                - Outputs Probabilities & Class
                 │                                                │
                 └───────────────────────┬────────────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  Ergonomic HUD & Feedback Engine     │
                      │   - Live Screen Status (Good / Bad)  │
                      │   - Non-blocking Audio Speech Alerts │
                      │   - Real-time Posture Score (0-100%) │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      Continuous Learning Loop        │
                      │   - Saves User Poses to CSV Dataset  │
                      │   - Online partial_fit / Retraining  │
                      │   - Model updates with each user!    │
                      └──────────────────────────────────────┘
```

### 1. 38-Dimensional Scale-Invariant Feature Extraction
Raw coordinates change whenever the user leans closer or moves further from the camera. MoveSure normalizes all measurements using the **shoulder span vector**:
- **Shoulder Tilt Angle**: Exact angle of the shoulder vector relative to horizontal.
- **Normalized Shoulder Delta**: `(left_shoulder.y - right_shoulder.y) / shoulder_span`.
- **Cervical / Neck Flexion**: Triangulation between ear, shoulder, and vertical axis.
- **Trunk Angle**: Angle between shoulder, hip, and vertical axis.
- **Head-to-Shoulder Height Ratio**: Vertical distance between nose and shoulder midpoint divided by shoulder width.
- **Forward Head Displacement**: Horizontal delta between ear center and shoulder midpoint.
- **Normalized Upper-Body Coordinate Offsets**: Relative $(x, y)$ positions of nose, eyes, ears, shoulders, elbows, wrists, and hips.

### 2. Posture Taxonomy & Ergonomic Signatures

| Class ID | Posture State | Ergonomic Signature & Clinical Indicators | System Feedback |
| :---: | :--- | :--- | :--- |
| **0** | **Good Posture** | Neck angle 10–18°, balanced shoulders ($\Delta < 2^\circ$), upright spine | *"Good posture! Keep it up."* |
| **1** | **Slouching** | Collapsed spine, torso compression ratio $< 1.25$, sunken shoulders | *"Slouching detected! Sit upright and engage core."* |
| **2** | **Forward Head** | Anterior cervical projection, forward head offset $> 0.15$, neck $> 28^\circ$ | *"Forward head tilt! Pull chin back to align neck."* |
| **3** | **Uneven Shoulders** | Lateral shoulder asymmetry, tilt angle $> 6^\circ$ | *"Shoulders misaligned! Level your shoulders evenly."* |

### 3. Model-Driven Adaptive Calibration
Every body is proportioned differently: neck length, torso height, desk setup, and webcam placement vary from person to person. 
* Pressing **'C'** (or calling `/calibrate`) records the user sitting naturally for 60 frames (3 seconds).
* The system computes their personal landmark distribution $(\boldsymbol{\mu}_{\text{user}}, \boldsymbol{\sigma}_{\text{user}})$.
* Rather than applying fixed hardcoded rules, the model computes an **adaptive residual offset** against ideal clinical ergonomics:
  $$\mathbf{f}_{\text{calibrated}} = \mathbf{f}_{\text{raw}} - (\boldsymbol{\mu}_{\text{user}} - \boldsymbol{\mu}_{\text{ideal}})$$
* The calibrated profile is saved to `backend/models/calibration_profile.json` for persistence.

### 4. Continuous Model Training With Every New User
When new users test the application:
* They can record labeled samples using the **User Training Studio** (`python -m backend.models.collect_posture_data`).
* Samples are automatically recorded to `data/user_posture_dataset.csv`.
* The classifier immediately updates its weights via incremental learning (`partial_fit`) or retrains the ensemble.
* The model continuously saves its updated state to `backend/models/posture_model.joblib`.

---

## 📂 Project Structure

```
MoveSure/
├── backend/
│   ├── app.py                      # Flask web server (streaming, calibration, & training API)
│   ├── models/
│   │   ├── feature_extractor.py    # 38-dim scale-invariant biomechanical feature engine
│   │   ├── dataset_generator.py    # Foundation biomechanical dataset synthesizer
│   │   ├── train_model.py          # Training & cross-validation script (SGD & Random Forest)
│   │   ├── posture_classifier.py   # Core ML classifier, adaptive calibration & online learning
│   │   └── collect_posture_data.py # Interactive webcam studio for user testing & training
│   └── utils/
│       └── movement_detection.py   # MediaPipe Pose + OpenCV HUD + non-blocking speech alerts
├── data/                           # User & baseline training datasets (.csv)
├── static/                         # Frontend styling, CSS, illustrations
│   ├── home.css
│   ├── login.css
│   └── register.css
├── templates/                      # Jinja2 / HTML templates
│   ├── home.html                   # Analytics dashboard & camera controls
│   ├── login.html                  # User login interface
│   └── register.html               # User registration interface
├── app.py                          # Full-stack Flask app with secure auth & dashboard
├── app.js                          # Node.js Express server with bcrypt & MongoDB
├── main.py                         # Desktop AI vision application
├── test.py                         # Test runner & quick launcher
├── train.py                        # Model training CLI shortcut
├── requirements.txt                # Python package dependencies
├── package.json                    # Node.js dependencies
├── .env.example                    # Environment variable configuration template
└── .gitignore                      # Git exclusion rules
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.9 or higher
- **Node.js**: 16+ (optional, for Node.js API server)
- **Webcam**: Built-in or external USB camera

### 2. Clone the Repository
```bash
git clone https://github.com/TayyabNadeem1/MoveSure.git
cd MoveSure
```

### 3. Environment Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Key configuration parameters in `.env`:
```ini
CAMERA_INDEX=0
POSTURE_MODEL_PATH=backend/models/posture_model.joblib
AUDIO_FEEDBACK_ENABLED=true
AUDIO_COOLDOWN_SECONDS=4
SECRET_KEY=your-secret-key-here
PORT=3000
FLASK_PORT=5000
```

### 4. Install Python Dependencies
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 5. (Optional) Install Node.js Dependencies
```bash
npm install
```

---

## 🚀 Running the Applications

### Option A: Desktop AI Vision Application (Recommended)
Runs the standalone OpenCV window with real-time HUD overlays, angles readout, and voice reminders:
```bash
python main.py
```
**Keyboard Controls:**
* **`C`**: Calibrate the model to your neutral sitting posture (3 seconds).
* **`Q`**: Exit the application.

---

### Option B: Flask Full-Stack Web Dashboard
Runs the web application featuring login, user registration, posture analytics charts, and live camera feed:
```bash
python app.py
```
Open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

### Option C: User Training Studio (Train the Model as a New User)
Sit in front of your webcam, record custom posture samples, and immediately update the machine learning model:
```bash
python -m backend.models.collect_posture_data --user john_doe
```
**Interactive Studio Keys:**
* **`0`**: Record Good Posture samples (60 frames)
* **`1`**: Record Slouching samples (60 frames)
* **`2`**: Record Forward Head samples (60 frames)
* **`3`**: Record Uneven Shoulders samples (60 frames)
* **`C`**: Calibrate neutral sitting posture
* **`T`**: Train & update the machine learning model with your newly recorded poses!
* **`Q`**: Quit studio.

---

### Option D: Node.js API Server
Runs the Express authentication server:
```bash
npm start
```

---

## 🎓 Model Training & Evaluation

To train or retrain the posture model using the command-line interface:

```bash
# Train online incremental learner (SGDClassifier with log_loss)
python train.py --model sgd

# Train ensemble Random Forest Classifier (100 estimators)
python train.py --model rf

# Train using a custom or combined user CSV dataset
python train.py --data data/user_posture_dataset.csv
```

### Sample Evaluation Metrics
```
============================================================
[*] Model Evaluation Accuracy: 98.25%
============================================================
                   precision    recall  f1-score   support

     Good Posture       0.99      0.98      0.98       160
        Slouching       0.97      0.99      0.98       160
     Forward Head       0.98      0.97      0.98       160
 Uneven Shoulders       0.99      0.99      0.99       160

         accuracy                           0.98       640
        macro avg       0.98      0.98      0.98       640
     weighted avg       0.98      0.98      0.98       640
```

---

## 🌐 Web API Reference

The Flask backend exposes RESTful endpoints for integrating with frontends:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/video_feed` | `GET` | Multipart MJPEG live video stream with ML posture HUD overlays |
| `/calibrate` | `POST` | Triggers a 60-frame model-based calibration session for the user |
| `/train-user` | `POST` | Receives JSON labeled posture vectors to incrementally update model weights |
| `/posture-status`| `GET` | Returns live posture stats, percentage of good/bad posture, angles, and score |
| `/start-posture` | `POST` | Launches the posture detection engine |
| `/stop-posture`  | `POST` | Stops active posture detection process |

---

## 🔒 Security & Privacy

* **Local Inference**: All computer vision processing and MediaPipe inferences occur entirely on your local machine; video frames are never transmitted to external servers.
* **Encrypted Passwords**: Authentication utilizes `werkzeug.security` (Python) and `bcryptjs` (Node.js) with salted password hashes.
* **Clean Git Hygiene**: Virtual environments, secrets, databases, and temporary audio files are strictly excluded via `.gitignore`.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, feedback, and user-trained ergonomic datasets are warmly welcomed! Please feel free to open an issue or submit a pull request.
