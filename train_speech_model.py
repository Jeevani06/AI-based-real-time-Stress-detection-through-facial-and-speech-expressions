"""
Training Script for Speech Emotion Recognition Model
Trains on RAVDESS dataset organized by emotion
"""

import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURATION
# =====================================================================
RAVDESS_ORGANIZED = r"C:\Project\stress_detection_project\data\RAVDESS\organized_by_emotion"
SAVED_MODELS_DIR  = r"C:\Project\stress_detection_project\saved_models"
EPOCHS      = 100
BATCH_SIZE  = 32
SAMPLE_RATE = 22050
DURATION    = 3      # seconds to use from each file
N_MFCC      = 40     # number of MFCC features

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def extract_features(file_path, sample_rate=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)

        # Pad if audio is shorter than duration
        target_length = sample_rate * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std  = np.std(mfcc,  axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel, axis=1)[:20]   # use first 20 bands

        # ZCR & RMS
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))

        features = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean, [zcr, rms]])
        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# =====================================================================
# LOAD DATASET
# =====================================================================
def load_dataset(data_dir):
    print("\nLoading and extracting features from RAVDESS dataset...")
    X, y = [], []
    total = 0

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"  ⚠️  Folder not found: {emotion_dir}")
            continue

        files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
        print(f"  Processing {emotion:10s}: {len(files)} files")

        for fname in files:
            fpath = os.path.join(emotion_dir, fname)
            features = extract_features(fpath)
            if features is not None:
                X.append(features)
                y.append(emotion)
                total += 1

    print(f"\nTotal samples extracted: {total}")
    return np.array(X), np.array(y)


# =====================================================================
# BUILD MODEL
# =====================================================================
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "="*70)
    print("SPEECH MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    return model


# =====================================================================
# PLOT HISTORY
# =====================================================================
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Speech Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Speech Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(SAVED_MODELS_DIR), 'speech_training_history.png')
    plt.savefig(save_path)
    print(f"\nTraining history plot saved to {save_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("="*70)
    print("SPEECH EMOTION RECOGNITION MODEL TRAINING")
    print("="*70)

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # Load data
    X, y_raw = load_dataset(RAVDESS_ORGANIZED)

    if len(X) == 0:
        print("\n❌ No data found! Check your RAVDESS organized path.")
        return

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded, num_classes=len(le.classes_))

    print(f"\nClasses found: {list(le.classes_)}")
    print(f"Feature vector size: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and label encoder for inference
    scaler_path = os.path.join(SAVED_MODELS_DIR, 'speech_scaler.pkl')
    le_path     = os.path.join(SAVED_MODELS_DIR, 'speech_label_encoder.pkl')
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    with open(le_path,     'wb') as f: pickle.dump(le,     f)
    print(f"\nScaler saved to:        {scaler_path}")
    print(f"Label encoder saved to: {le_path}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTraining samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Build model
    model = build_model(X_train.shape[1], len(le.classes_))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            os.path.join(SAVED_MODELS_DIR, 'best_speech_model.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_path = os.path.join(SAVED_MODELS_DIR, 'speech_emotion_model.h5')
    model.save(final_path)
    print(f"\n✅ Final model saved to: {final_path}")

    # Plot
    plot_history(history)

    # Final accuracy
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Models saved in: {SAVED_MODELS_DIR}")
    print(f"\nFiles saved:")
    print(f"  - speech_emotion_model.h5")
    print(f"  - best_speech_model.h5")
    print(f"  - speech_scaler.pkl")
    print(f"  - speech_label_encoder.pkl")
    print(f"\n✅ You can now run: python stress_detection_system.py")


if __name__ == "__main__":
    main()