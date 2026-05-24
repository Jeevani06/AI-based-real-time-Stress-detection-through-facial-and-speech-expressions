"""
MindScan — Flask Backend (FIXED)
- Removed duplicate DURATION/SAMPLE_RATE definitions
- Fixed feature vector to match training exactly
- Added feature dimension validation
- Added debug logging for speech predictions
- Improved audio decoding with fallback chain
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import pickle
import librosa
import soundfile as sf
import io
import os
import subprocess
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "mindscan-secret-key-2024"
CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "null",
    ],
)

# -------------------------------------------------
# LOAD MODELS AT STARTUP
# -------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Face emotion model (FER2013, 7 classes)
face_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_model.h5"))
FACE_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Speech emotion model + scaler + label encoder (RAVDESS, 8 classes)
speech_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "speech_emotion_model.h5"))
with open(os.path.join(MODELS_DIR, "speech_scaler.pkl"), "rb") as f:
    speech_scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "speech_label_encoder.pkl"), "rb") as f:
    speech_le = pickle.load(f)

# OpenCV Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# In-memory user store  {username: hashed_password}
users_db = {
    "demo":    generate_password_hash("Demo@123"),
    "kavya":   generate_password_hash("Kavya@123"),
    "jeevani": generate_password_hash("Jeevani@123"),
    "sushma":  generate_password_hash("Sushma@123"),
}

# -------------------------------------------------
# AUDIO / FEATURE CONSTANTS  (single definition)
# -------------------------------------------------
SAMPLE_RATE = 22050
DURATION    = 10      # seconds — must match your recording length
N_MFCC      = 40

# Print model info at startup to help debug
print("✅ All models loaded successfully.")
print(f"✅ Speech label order: {list(speech_le.classes_)}")
print(f"✅ Speech model input shape: {speech_model.input_shape}")
print(f"✅ Scaler n_features_in_: {getattr(speech_scaler, 'n_features_in_', 'unknown')}")

# -------------------------------------------------
# STRESS WEIGHT TABLE
# -------------------------------------------------
STRESS_WEIGHTS = {
    "angry":     0.9,
    "disgust":   0.8,
    "fear":      0.9,
    "fearful":   0.9,
    "sad":       0.7,
    "surprise":  0.5,
    "surprised": 0.5,
    "neutral":   0.2,
    "happy":     0.1,
    "calm":      0.1,
}

def compute_stress(probabilities, labels):
    """Weighted stress score 0-100."""
    score = sum(
        prob * STRESS_WEIGHTS.get(label.lower(), 0)
        for prob, label in zip(probabilities, labels)
    )
    return round(score * 100, 1)

def stress_level(score):
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "low"

# -------------------------------------------------
# AUTH ROUTES
# -------------------------------------------------

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.json or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "message": "Missing credentials"}), 400

    hashed = users_db.get(username)
    if hashed and check_password_hash(hashed, password):
        session["user"] = username
        return jsonify({"success": True, "username": username})
    return jsonify({"success": False, "message": "Invalid username or password"}), 401


@app.route("/api/signup", methods=["POST"])
def signup():
    data     = request.json or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "message": "Missing fields"}), 400
    if username in users_db:
        return jsonify({"success": False, "message": "Username already taken"}), 409

    users_db[username] = generate_password_hash(password)
    session["user"] = username
    return jsonify({"success": True, "username": username})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"success": True})

# -------------------------------------------------
# FACE EMOTION PREDICTION
# POST /api/predict/face
# Body: { "image": "data:image/jpeg;base64,..." }
# -------------------------------------------------

@app.route("/api/predict/face", methods=["POST"])
def predict_face():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"face_detected": False, "error": "No image"}), 400

        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_bytes = base64.b64decode(img_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"face_detected": False, "error": "Invalid image"}), 400

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return jsonify({"face_detected": False})

        x, y, w, h    = max(faces, key=lambda r: r[2] * r[3])
        face_roi       = gray[y: y + h, x: x + w]
        face_resized   = cv2.resize(face_roi, (48, 48))
        face_norm      = face_resized.astype("float32") / 255.0
        face_input     = face_norm.reshape(1, 48, 48, 1)

        preds      = face_model.predict(face_input, verbose=0)[0]
        idx        = int(np.argmax(preds))
        confidence = float(preds[idx]) * 100

        return jsonify({
            "face_detected": True,
            "emotion":       FACE_LABELS[idx],
            "confidence":    round(confidence, 1),
            "probabilities": preds.tolist(),
            "labels":        FACE_LABELS,
            "face_box":      [int(x), int(y), int(w), int(h)],
        })

    except Exception as e:
        print(f"[predict_face] Error: {e}")
        return jsonify({"face_detected": False, "error": str(e)}), 500

# -------------------------------------------------
# AUDIO DECODE  (webm/opus → float32 mono waveform)
# -------------------------------------------------

def _decode_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw audio bytes to a float32 mono waveform at SAMPLE_RATE.
    Tries soundfile first, then ffmpeg pipe.
    """
    # Attempt 1: soundfile (works for wav/ogg/flac)
    try:
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"[audio] soundfile decode OK — sr={sr}, samples={len(audio)}, duration={len(audio)/SAMPLE_RATE:.2f}s")
        return audio
    except Exception as e:
        print(f"[audio] soundfile failed ({e}), trying ffmpeg…")

    # Attempt 2: ffmpeg pipe (handles webm/opus from browser)
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-f", "f32le",
                "pipe:1",
            ],
            input=raw_bytes,
            capture_output=True,
            timeout=20,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {proc.stderr.decode()}")
        audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        print(f"[audio] ffmpeg decode OK — samples={len(audio)}, duration={len(audio)/SAMPLE_RATE:.2f}s")
        if len(audio) == 0:
            raise RuntimeError("ffmpeg produced 0 samples — is the audio valid?")
        return audio
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it:\n"
            "  Windows: https://ffmpeg.org/download.html  (add to PATH)\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Mac:     brew install ffmpeg"
        )

# -------------------------------------------------
# FEATURE EXTRACTION
# Must exactly match what was used during training.
# -------------------------------------------------

def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract feature vector. Pads/trims to DURATION seconds.

    Feature layout (must match training):
      MFCC mean     : N_MFCC values  (40)
      MFCC std      : N_MFCC values  (40)
      Chroma mean   : 12 values
      Mel mean      : 20 values (first 20 mel bands)
      ZCR mean      : 1 value
      RMS mean      : 1 value
      Total         : 114 values
    """
    target_len = sr * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Normalise amplitude to avoid near-silence issues
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val

    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean   = np.mean(mfcc, axis=1)          # (40,)
    mfcc_std    = np.std(mfcc,  axis=1)          # (40,)

    chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)        # (12,)

    mel         = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean    = np.mean(mel, axis=1)[:20]      # (20,)

    zcr  = np.array([np.mean(librosa.feature.zero_crossing_rate(audio))])  # (1,)
    rms  = np.array([np.mean(librosa.feature.rms(y=audio))])               # (1,)

    features = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean, zcr, rms])
    print(f"[features] vector size = {features.shape[0]}")
    return features


# -------------------------------------------------
# SPEECH EMOTION PREDICTION
# POST /api/predict/speech
# Body: multipart/form-data  field = "audio"
# -------------------------------------------------

@app.route("/api/predict/speech", methods=["POST"])
def predict_speech():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio field in request"}), 400

        raw_bytes = request.files["audio"].read()
        if not raw_bytes:
            return jsonify({"error": "Empty audio upload"}), 400

        print(f"[predict_speech] Received {len(raw_bytes)} bytes of audio")

        audio = _decode_audio_bytes(raw_bytes)

        features = extract_features(audio)

        # ── CRITICAL: validate feature dimension against scaler ──
        expected_features = getattr(speech_scaler, 'n_features_in_', None)
        if expected_features and features.shape[0] != expected_features:
            msg = (
                f"Feature mismatch: model expects {expected_features} features "
                f"but extract_features() produced {features.shape[0]}. "
                f"Edit extract_features() to match your training script exactly."
            )
            print(f"[predict_speech] ERROR — {msg}")
            return jsonify({"error": msg}), 500

        features_scaled = speech_scaler.transform(features.reshape(1, -1))

        preds      = speech_model.predict(features_scaled, verbose=0)[0]
        idx        = int(np.argmax(preds))
        label      = speech_le.classes_[idx]
        confidence = float(preds[idx]) * 100

        speech_labels = list(speech_le.classes_)

        # Debug: print all probabilities so you can see if model is stuck
        prob_str = ", ".join(f"{l}:{p*100:.1f}%" for l, p in zip(speech_labels, preds))
        print(f"[predict_speech] → {label} ({confidence:.1f}%) | {prob_str}")

        probs_dict = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(speech_labels, preds)
        }

        return jsonify({
            "emotion":       label,
            "confidence":    round(confidence, 1),
            "probabilities": preds.tolist(),
            "labels":        speech_labels,
            "probs_dict":    probs_dict,
        })

    except RuntimeError as e:
        print(f"[predict_speech] Decode error: {e}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[predict_speech] Error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# FINAL STRESS ANALYSIS
# POST /api/analyze
# -------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}

        face_emotion   = data.get("face_emotion")   or "Neutral"
        face_probs     = data.get("face_probs")      or []
        speech_emotion = data.get("speech_emotion") or "neutral"
        speech_probs   = data.get("speech_probs")   or []

        speech_labels_from_req = data.get("speech_labels") or []
        speech_labels = (
            speech_labels_from_req
            if len(speech_labels_from_req) == len(speech_le.classes_)
            else list(speech_le.classes_)
        )

        # Face stress score
        if face_probs and len(face_probs) == len(FACE_LABELS):
            f_score = compute_stress(face_probs, FACE_LABELS)
        else:
            f_score = STRESS_WEIGHTS.get(face_emotion.lower(), 0.4) * 100

        # Speech stress score
        if speech_probs and len(speech_probs) == len(speech_labels):
            s_score = compute_stress(speech_probs, speech_labels)
        else:
            s_score = STRESS_WEIGHTS.get(speech_emotion.lower(), 0.4) * 100

        print(f"[analyze] face_score={f_score} speech_score={s_score} "
              f"face_emo={face_emotion} speech_emo={speech_emotion}")

        combined = round(f_score * 0.6 + s_score * 0.4, 1)
        level    = stress_level(combined)

        # Build distributions for charts
        face_dist = {}
        if face_probs and len(face_probs) == len(FACE_LABELS):
            for i, lbl in enumerate(FACE_LABELS):
                face_dist[lbl] = round(float(face_probs[i]) * 100, 2)
        else:
            face_dist = {lbl: 0 for lbl in FACE_LABELS}
            face_dist[face_emotion] = 100

        speech_dist = {}
        if speech_probs and len(speech_probs) == len(speech_labels):
            for i, lbl in enumerate(speech_labels):
                speech_dist[lbl] = round(float(speech_probs[i]) * 100, 2)
        else:
            speech_dist = {lbl: 0 for lbl in speech_labels}
            speech_dist[speech_emotion] = 100

        return jsonify({
            "stress_score":   combined,
            "stress_level":   level,
            "face_score":     round(f_score, 1),
            "speech_score":   round(s_score, 1),
            "face_emotion":   face_emotion,
            "speech_emotion": speech_emotion,
            "face_dist":      face_dist,
            "speech_dist":    speech_dist,
            "speech_labels":  speech_labels,
            "timestamp":      datetime.now().strftime("%d %b %Y, %I:%M %p"),
        })

    except Exception as e:
        print(f"[analyze] Error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# DEBUG ENDPOINT — call this to check feature size
# GET /api/debug/speech
# -------------------------------------------------

@app.route("/api/debug/speech", methods=["GET"])
def debug_speech():
    """Returns model config to help diagnose feature mismatches."""
    return jsonify({
        "speech_labels":          list(speech_le.classes_),
        "n_labels":               len(speech_le.classes_),
        "model_input_shape":      str(speech_model.input_shape),
        "scaler_n_features":      getattr(speech_scaler, 'n_features_in_', 'unknown'),
        "extract_features_size":  114,  # N_MFCC*2 + 12 + 20 + 2 = 114
        "sample_rate":            SAMPLE_RATE,
        "duration":               DURATION,
        "n_mfcc":                 N_MFCC,
    })

# -------------------------------------------------
# ROOT
# -------------------------------------------------

@app.route("/")
def home():
    return "MindScan Backend Running ✅ (fixed speech edition)"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
