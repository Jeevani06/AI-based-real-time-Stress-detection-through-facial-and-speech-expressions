
"""
Facial Emotion Recognition Module - FIXED VERSION
Detects and classifies emotions from facial expressions
Compatible with TensorFlow 2.x and Python 3.11
"""

import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

class FacialEmotionRecognizer:
    """
    Real-time facial emotion recognition using CNN
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognizer
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.img_size = 48
        
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = self.build_model()
            print("New model created")
    
    def build_model(self):
        """
        Build CNN architecture for emotion recognition - FIXED VERSION
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer
            Input(shape=(self.img_size, self.img_size, 1)),
            
            # First Convolution Block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolution Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolution Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth Convolution Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        model.summary()
        
        return model
    
    def train_model(self, train_dir, val_dir, epochs=50, batch_size=32):
        """
        Train the emotion recognition model
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            classes=self.emotions
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            classes=self.emotions
        )
        
        # Create saved_models directory
        os.makedirs('saved_models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint('saved_models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # Train model
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face coordinates (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    import cv2

class FacialEmotionRecognizer:

    def __init__(self, model_path):
        self.model = load_model(model_path)
        
        # ✅ Add this line
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 🚨 IMPORTANT CHECK
        if len(faces) == 0:
            return "No face detected"

        # Take first face
        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]

        # preprocess and predict
        processed = self.preprocess(face)
        prediction = self.model.predict(processed)

        emotion = self.get_label(prediction)

        return emotion
    
    def process_frame(self, frame):
        """
        Process a single frame for emotion detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with annotations and emotion data
        """
        faces = self.detect_faces(frame)
        emotion_data = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidences = self.predict_emotion(face_roi)
            emotion_data.append({
                'emotion': emotion,
                'confidences': confidences,
                'bbox': (x, y, w, h)
            })
            
            # Color based on emotion
            colors = {
                'Angry': (0, 0, 255),
                'Disgust': (128, 0, 128),
                'Fear': (0, 100, 255),
                'Happy': (0, 255, 0),
                'Sad': (255, 100, 0),
                'Surprise': (255, 0, 255),
                'Neutral': (255, 255, 0)
            }
            color = colors.get(emotion, (255, 255, 255))
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display emotion label
            label = f"{emotion}: {confidences[self.emotions.index(emotion)]:.2f}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame, emotion_data
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")

def main():
    """
    Demo function for testing facial emotion recognition
    """
    print("="*70)
    print("FACIAL EMOTION RECOGNITION - TensorFlow Version")
    print("="*70)
    print("\nInitializing...")
    
    recognizer = FacialEmotionRecognizer()
    
    print("\n⚠️  NOTE: This model is NOT trained yet!")
    print("It will show random predictions until you train it.")
    print("\nTo train the model, run:")
    print("  python train_facial_model.py")
    print("\nPress 'q' to quit\n")
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        print("Trying alternative camera...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ No camera found!")
            return
    
    print("✅ Camera opened successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, emotion_data = recognizer.process_frame(frame)
        
        # Add warning text
        cv2.putText(processed_frame, "UNTRAINED MODEL - Random predictions", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(processed_frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display results
        cv2.imshow('Facial Emotion Recognition', processed_frame)
        
        # Print emotion data
        if emotion_data:
            for data in emotion_data:
                print(f"\rDetected: {data['emotion']:10s}", end='', flush=True)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n\n✅ Demo completed!")

if __name__ == "__main__":
    main()

