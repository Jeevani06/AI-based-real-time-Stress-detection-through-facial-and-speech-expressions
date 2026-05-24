# MindScan — Setup & Run Guide

## Folder Structure (place inside your project)
```
stress_detection_project/
├── app.py                  ← Flask backend (NEW)
├── static/
│   └── index.html          ← Beautiful frontend (NEW)
├── requirements_app.txt    ← New dependencies (NEW)
├── models/
│   ├── facial_emotion_recognition.py
│   └── speech_emotion_recognition.py
├── saved_models/
│   ├── facial_emotion_model.h5
│   ├── speech_emotion_model.h5
│   ├── speech_scaler.pkl
│   └── speech_label_encoder.pkl
└── ...
```

## Step 1 — Install Flask dependencies
```bash
pip install flask flask-cors werkzeug
```
(TensorFlow, OpenCV, Librosa are already installed from your training setup)

## Step 2 — Verify model paths in app.py
Open `app.py` and confirm these paths match YOUR system:
```python
BASE_DIR   = r"C:\Project\stress_detection_project"
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
```

## Step 3 — Run the server
```bash
cd C:\Project\stress_detection_project
python app.py
```
You should see:
```
✅ Facial model loaded from ...
✅ Speech model loaded from ...
🚀 MindScan server starting at http://127.0.0.1:5000
```

## Step 4 — Open in browser
Go to: http://127.0.0.1:5000

## Login Credentials
| Username | Password    |
|----------|-------------|
| kavya    | kavya123    |
| jeevani  | jeevani123  |
| sushma   | sushma123   |
| demo     | demo123     |

To add users, edit the USERS dict in app.py.

## How It Works
1. **Login** → Session created
2. **Camera** starts via browser WebRTC
3. **Start Analysis** button:
   - Captures 5 webcam frames → sends as base64 to `/api/predict/face`
   - Backend runs your CNN model (facial_emotion_model.h5)
   - Records 5 seconds of audio → sends to `/api/predict/speech`
   - Backend extracts MFCC features (same as your training script) → runs Dense NN
   - Calls `/api/analyze` to fuse results (60% face, 40% speech)
4. **Report** shows:
   - Animated stress gauge (0–100%)
   - Pie charts for facial + speech emotion distribution
   - Motivational quotes
   - YouTube exercise / meditation / music recommendations
   - Based on stress level: High / Medium / Low

## Offline / Demo Mode
If models aren't loaded yet, the app automatically uses realistic
random demo data so you can test the full UI flow.

## Troubleshooting
- **Camera not working**: Allow camera permissions in browser
- **Model not loading**: Check saved_models/ paths in app.py
- **PyAudio error**: Audio recording uses browser WebRTC (no PyAudio needed!)
- **CORS error**: Make sure you're accessing via http://127.0.0.1:5000 not file://
