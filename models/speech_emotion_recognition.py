"""
Speech Emotion Recognition Module
Analyzes audio input to detect emotional states
"""

import librosa
import numpy as np
import soundfile as sf
import pyaudio
import wave
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os

class SpeechEmotionRecognizer:
    """
    Real-time speech emotion recognition
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the speech emotion recognizer
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.scaler = StandardScaler()
        
        # Audio recording parameters
        self.chunk_size = 1024
        self.sample_rate = 22050
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Speech model loaded from {model_path}")
        else:
            self.model = self.build_model()
            print("New speech model created")
    
    def build_model(self, input_shape=180):
        """
        Build LSTM model for speech emotion recognition
        
        Args:
            input_shape: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(256, return_sequences=True, input_shape=(input_shape, 1)),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(128),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_features(self, audio_path=None, audio_data=None, sr=22050):
        """
        Extract audio features for emotion recognition
        
        Args:
            audio_path: Path to audio file (optional)
            audio_data: Audio data array (optional)
            sr: Sample rate
            
        Returns:
            Feature vector
        """
        # Load audio
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=sr, duration=3)
        elif audio_data is not None:
            audio = audio_data
        else:
            raise ValueError("Either audio_path or audio_data must be provided")
        
        # Extract features
        features = []
        
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        features.extend(mfcc_mean)
        
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        features.extend(chroma_mean)
        
        # 3. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        features.extend(mel_mean)
        
        # 4. Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        features.extend(contrast_mean)
        
        # 5. Tonnetz
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        features.extend(tonnetz_mean)
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr.T, axis=0)
        features.extend(zcr_mean)
        
        # 7. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.extend(np.mean(spectral_centroids.T, axis=0))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.extend(np.mean(spectral_rolloff.T, axis=0))
        
        return np.array(features)
    
    def predict_emotion(self, audio_path=None, audio_data=None):
        """
        Predict emotion from audio
        
        Args:
            audio_path: Path to audio file (optional)
            audio_data: Audio data array (optional)
            
        Returns:
            Tuple of (emotion_label, confidence_scores)
        """
        # Extract features
        features = self.extract_features(audio_path=audio_path, audio_data=audio_data)
        
        # Ensure correct feature length
        target_length = 180
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)), mode='constant')
        elif len(features) > target_length:
            features = features[:target_length]
        
        # Reshape for LSTM
        features = features.reshape(1, target_length, 1)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        return self.emotions[emotion_idx], predictions[0]
    
    def record_audio(self, duration=3, filename="temp_audio.wav"):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            filename: Output filename
            
        Returns:
            Path to recorded audio file
        """
        print(f"Recording for {duration} seconds...")
        
        audio = pyaudio.PyAudio()
        
        # Open stream
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        
        # Record
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        print("Recording complete!")
        
        # Stop stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save to file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return filename
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the speech emotion model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001),
            ModelCheckpoint('saved_models/best_speech_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
        print(f"Speech model saved to {path}")


def main():
    """
    Demo function for testing speech emotion recognition
    """
    print("Initializing Speech Emotion Recognizer...")
    recognizer = SpeechEmotionRecognizer()
    
    while True:
        input("Press Enter to record (3 seconds)...")
        
        # Record audio
        audio_file = recognizer.record_audio(duration=3)
        
        # Predict emotion
        emotion, confidences = recognizer.predict_emotion(audio_path=audio_file)
        
        print(f"\nDetected Emotion: {emotion}")
        print("Confidence scores:")
        for em, conf in zip(recognizer.emotions, confidences):
            print(f"  {em}: {conf:.4f}")
        
        # Ask to continue
        cont = input("\nContinue? (y/n): ")
        if cont.lower() != 'y':
            break
        
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)


if __name__ == "__main__":
    main()
