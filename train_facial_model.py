"""
Training Script for Facial Emotion Recognition Model
Train the CNN model on FER2013 or custom dataset
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.facial_emotion_recognition import FacialEmotionRecognizer

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

def prepare_fer2013_dataset(csv_path):
    """
    Prepare FER2013 dataset from CSV file
    
    Args:
        csv_path: Path to fer2013.csv file
        
    Returns:
        Train and validation data
    """
    import pandas as pd
    
    print("Loading FER2013 dataset...")
    df = pd.read_csv(csv_path)
    
    # Extract pixels and emotions
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].tolist()
    usage = df['Usage'].tolist()
    
    # Convert pixels to arrays
    X = []
    y = []
    
    for pixel_string, emotion, use in zip(pixels, emotions, usage):
        pixel_array = np.array([int(p) for p in pixel_string.split()], dtype='uint8')
        pixel_array = pixel_array.reshape(48, 48, 1)
        X.append(pixel_array)
        y.append(emotion)
    
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    # One-hot encode labels
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=7)
    
    # Split by usage
    X_train = X[[i for i, u in enumerate(usage) if u == 'Training']]
    y_train = y[[i for i, u in enumerate(usage) if u == 'Training']]
    
    X_val = X[[i for i, u in enumerate(usage) if u in ['PublicTest', 'PrivateTest']]]
    y_val = y[[i for i, u in enumerate(usage) if u in ['PublicTest', 'PrivateTest']]]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val

def train_from_directory(train_dir, val_dir, epochs=50, batch_size=32):
    """
    Train model from directory structure
    
    Directory structure should be:
    train_dir/
        Angry/
        Disgust/
        Fear/
        Happy/
        Sad/
        Surprise/
        Neutral/
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        epochs: Number of epochs
        batch_size: Batch size
    """
    print("Training from directory structure...")
    
    # Initialize recognizer
    recognizer = FacialEmotionRecognizer()
    
    # Train model
    history = recognizer.train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save model
    recognizer.save_model('saved_models/facial_emotion_model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    return recognizer, history

def train_from_csv(csv_path, epochs=50, batch_size=32):
    """
    Train model from FER2013 CSV file
    
    Args:
        csv_path: Path to fer2013.csv
        epochs: Number of epochs
        batch_size: Batch size
    """
    print("Training from CSV file...")
    
    # Prepare dataset
    X_train, y_train, X_val, y_val = prepare_fer2013_dataset(csv_path)
    
    # Initialize recognizer
    recognizer = FacialEmotionRecognizer()
    
    # Train using fit method
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        ModelCheckpoint('saved_models/best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    history = recognizer.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    recognizer.save_model('saved_models/facial_emotion_model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    return recognizer, history

def main():
    """Main training function"""
    print("=" * 70)
    print("FACIAL EMOTION RECOGNITION MODEL TRAINING")
    print("=" * 70)
    
    # Create saved_models directory
    os.makedirs('saved_models', exist_ok=True)
    
    print("\nChoose training method:")
    print("1. Train from directory structure")
    print("2. Train from FER2013 CSV file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        train_dir = input("Enter training directory path: ").strip()
        val_dir = input("Enter validation directory path: ").strip()
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("Error: Directory does not exist!")
            return
        
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        batch_size = int(input("Enter batch size (default 32): ") or "32")
        
        recognizer, history = train_from_directory(
            train_dir, val_dir, epochs, batch_size
        )
        
    elif choice == '2':
        csv_path = input("Enter path to fer2013.csv: ").strip()
        
        if not os.path.exists(csv_path):
            print("Error: CSV file does not exist!")
            return
        
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        batch_size = int(input("Enter batch size (default 32): ") or "32")
        
        recognizer, history = train_from_csv(csv_path, epochs, batch_size)
    
    else:
        print("Invalid choice!")
        return
    
    print("\n✅ Training completed successfully!")
    print("Model saved to: saved_models/facial_emotion_model.h5")

if __name__ == "__main__":
    main()
