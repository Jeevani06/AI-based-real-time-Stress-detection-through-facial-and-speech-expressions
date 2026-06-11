# AI-based-real-time-Stress-detection-through-facial-and-speech-expressions
# MindScan — AI Based Real-Time Stress Detection

> Mini-Project 1 | Department of Computer Science & Engineering  
> G. Narayanamma Institute of Technology & Science (Autonomous), Hyderabad  
> Affiliated to JNTUH | June 2026

---

## Team

| Name | Roll No |
|---|---|
| A. Kavya Sri | 24251A05W5 |
| G. Jeevani | 24251A05X3 |
| J.V.S.S. Sushma | 24251A05X5 |

**Guide:** Mrs. G. Amulya, Assistant Professor, Dept. of CSE

---

## What This Project Does

**MindScan** is a browser-based AI application that detects your stress level in real time by simultaneously analysing your **face** (via webcam) and **voice** (via microphone) for 10 seconds. It then gives you a stress score (0–100%), classifies it as Low / Medium / High, and provides personalised wellness recommendations.

```
Webcam  →  Haar Cascade face detection  →  CNN (FER2013)  →  Facial emotion + confidence
Mic     →  Audio recording (10 s)       →  MFCC/Chroma/Mel features  →  Dense NN (RAVDESS)  →  Speech emotion + confidence
                                    ↓
                     Stress Score = 0.60 × Face + 0.40 × Speech
                                    ↓
                     LOW (< 40%) / MEDIUM (40–64%) / HIGH (≥ 65%)
                                    ↓
                     Personalised: Exercise · Meditation · Music
```

---

## Repository Files

```
├── app.py                   Flask backend (all API endpoints)
├── index.html               Single-page frontend (HTML + CSS + JS)
├── train_facial_model.py    CNN training script — FER2013
├── train_speech_model.py    Dense NN training script — RAVDESS
├── requirements_app.txt     Python dependencies
├── models/                  (model architecture helpers if any)
└── saved_models/            Trained model files (not in repo — see below)
    ├── best_model.h5
    ├── facial_emotion_model.h5
    ├── speech_emotion_model.h5
    ├── best_speech_model.h5
    ├── speech_scaler.pkl
    └── speech_label_encoder.pkl
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Jeevani06/AI-based-real-time-Stress-detection-through-facial-and-speech-expressions.git
cd AI-based-real-time-Stress-detection-through-facial-and-speech-expressions
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements_app.txt
```

> **Note (Windows audio):** If `soundfile` alone cannot decode browser WebM audio, install ffmpeg and add it to PATH:  
> Download from https://ffmpeg.org/download.html → add `bin/` folder to system PATH.

### 4. Train the models (first-time only)

#### Facial model — FER2013

Download `fer2013.csv` from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) and run:

```bash
python train_facial_model.py
# Choose option 1 → enter path to fer2013.csv
```

Outputs → `saved_models/best_model.h5` and `saved_models/facial_emotion_model.h5`

#### Speech model — RAVDESS

1. Download [RAVDESS](https://zenodo.org/record/1188976) audio files.
2. Organise into folders by emotion:

```
data/RAVDESS/organized_by_emotion/
    neutral/   calm/   happy/   sad/
    angry/     fearful/  disgust/  surprised/
```

3. Update `RAVDESS_DIR` in `train_speech_model.py` then run:

```bash
python train_speech_model.py
```

Outputs → `saved_models/speech_emotion_model.h5`, `speech_scaler.pkl`, `speech_label_encoder.pkl`

### 5. Start the Flask backend

```bash
python app.py
# Running on http://127.0.0.1:5000
```

### 6. Open the frontend

Open `index.html` in your browser (use VS Code Live Server on port 5500, or any local HTTP server). The frontend communicates with the backend at `http://127.0.0.1:5000`.

### 7. Log in and scan

Use one of the built-in demo accounts:

| Username | Password |
|---|---|
| demo | Demo@123 |
| kavya | Kavya@123 |
| jeevani | Jeevani@123 |
| sushma | Sushma@123 |

Click **▶ Start 10s Analysis**, allow camera and microphone access, stay still for 10 seconds, and see your results.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/login` | Authenticate user |
| POST | `/api/signup` | Register new user |
| POST | `/api/logout` | Clear session |
| POST | `/api/predict/face` | Detect face emotion from base64 image |
| POST | `/api/predict/speech` | Detect speech emotion from audio file |
| POST | `/api/analyze` | Compute weighted stress score |
| GET | `/api/debug/speech` | Show model config for debugging |

### `/api/predict/face`
```json
Request  : { "image": "data:image/jpeg;base64,..." }
Response : { "face_detected": true, "emotion": "Neutral",
             "confidence": 87.3, "probabilities": [...], "labels": [...] }
```

### `/api/predict/speech`
```
Request  : multipart/form-data  field="audio"  (WAV or WebM)
Response : { "emotion": "calm", "confidence": 72.1,
             "probabilities": [...], "labels": [...] }
```

### `/api/analyze`
```json
Request  : { "face_emotion": "Neutral", "face_probs": [...],
             "speech_emotion": "calm",  "speech_probs": [...] }
Response : { "stress_score": 18.4, "stress_level": "low",
             "face_score": 20.0, "speech_score": 10.0,
             "face_dist": {...}, "speech_dist": {...}, "timestamp": "..." }
```

---

## Models

### CNN — Facial Emotion Recognition (FER2013)

| Detail | Value |
|---|---|
| Dataset | FER2013 — 35,887 grayscale 48×48 images |
| Classes | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral (7) |
| Architecture | 4 Conv blocks (32→64→128→256) + BatchNorm + Flatten + Dense 512 → 128 → Softmax |
| Optimizer | Adam lr=0.001, Batch=64, Max epochs=50, Early stop patience=10 |
| Augmentation | Horizontal flip, ±15° rotation, brightness jitter |
| Test accuracy | ~65% |
| Inference | <5 ms/frame, CPU only |

### Dense NN — Speech Emotion Recognition (RAVDESS)

| Detail | Value |
|---|---|
| Dataset | RAVDESS — 1,440 WAV files, 24 actors, 48 kHz |
| Classes | neutral, calm, happy, sad, angry, fearful, disgust, surprised (8) |
| Feature vector | MFCC mean(40) + MFCC std(40) + Chroma(12) + Mel(20) + ZCR(1) + RMS(1) = **114 dims** |
| Architecture | Dense 512 → 256 → 128 → 64 → 32 → Softmax |
| Optimizer | Adam lr=0.001, Batch=32, Max epochs=100, Early stop patience=15 |
| Test accuracy | ~79.51% |
| Inference | <200 ms/clip, CPU only |

---

## Stress Fusion

```python
Stress Score = 0.60 × Face Score  +  0.40 × Speech Score

# Stress weights per emotion
STRESS_WEIGHTS = {
    "angry": 0.9,  "fearful": 0.9,  "fear": 0.9,
    "disgust": 0.8,
    "sad": 0.7,
    "surprise": 0.5,  "surprised": 0.5,
    "neutral": 0.2,
    "happy": 0.1,  "calm": 0.1,
}

# Classification thresholds
HIGH   = score >= 65
MEDIUM = 40 <= score < 65
LOW    = score < 40
```

---

## Performance Summary

| Approach | Accuracy | Real-Time | Recommendations |
|---|---|---|---|
| Face-only CNN | ~62% | ✅ | ❌ |
| Speech-only DNN | ~68% | ✅ | ❌ |
| Equal fusion (50-50) | ~71% | ✅ | ❌ |
| **Proposed (60-40)** | **~74%** | **✅** | **✅** |
| Sensor-based | ~85% | ✅ | ❌ |
| Cloud API (face only) | ~78% | ❌ | ❌ |

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| Flask | 3.0.0 | Backend REST API |
| Flask-CORS | 4.0.0 | Cross-origin requests |
| TensorFlow / Keras | 2.13.0 | CNN and Dense NN |
| OpenCV | 4.8.1 | Webcam + Haar Cascade face detection |
| Librosa | 0.10.1 | MFCC, Chroma, Mel Spectrogram extraction |
| SoundFile | 0.12.1 | Audio decoding (WAV/OGG/FLAC) |
| NumPy | 1.24.3 | Array operations |
| Scikit-learn | 1.3.0 | StandardScaler, LabelEncoder, metrics |
| HTML/CSS/JS | — | Frontend single-page app |
| Chart.js | CDN | Emotion distribution donut charts |

---

## Frontend Features (index.html)

- Login / Signup with password strength meter and lockout protection (5 failed attempts → 30 s lock)
- Live webcam feed with face bounding box and real-time emotion overlay
- Real-time audio visualiser bars during recording
- 10-second countdown ring over the video during scan
- Report page with gauge chart, emotion distribution charts, quote, and 3-tab recommendations
- Recommendation tabs: Exercise · Meditation · Music — each with YouTube links

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `best_model.h5` not found | Run `train_facial_model.py` first |
| `speech_emotion_model.h5` not found | Run `train_speech_model.py` first |
| Feature mismatch error | Ensure `extract_features()` in `app.py` and `train_speech_model.py` produce the same 114-dim vector |
| Audio always fails | Install ffmpeg and add to PATH |
| CORS error in browser | Ensure Flask runs on port 5000 and frontend is on 5500 |
| No face detected | Improve lighting, centre face, remove obstructions |
| Low speech accuracy | Reduce background noise, speak clearly into microphone |

---

## Future Enhancements

- Android / iOS mobile app
- Multilingual and accent-aware speech recognition
- Historical stress tracking dashboard
- Integrated wellness chatbot
- Larger, more diverse training datasets

---

## Datasets

| Dataset | Source |
|---|---|
| FER2013 | https://www.kaggle.com/datasets/msambare/fer2013 |
| RAVDESS | https://zenodo.org/record/1188976 |

---

## SDG Alignment

- **SDG 3** — Good Health and Well-Being
- **SDG 8** — Decent Work and Economic Growth
- **SDG 9** — Industry, Innovation, and Infrastructure

---

*Submitted in partial fulfillment of B.Tech (CSE) under JNTUH — June 2026*
