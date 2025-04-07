import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, data_path, sample_rate=22050, duration=30):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.genres = None
        self.label_encoder = LabelEncoder()

    def extract_features(self, file_path):
        """Extract features from an audio file."""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, duration=self.duration, sr=self.sample_rate)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Calculate statistics
            features = []
            for feature in [mfccs, spectral_centroid, chroma]:
                features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.max(feature),
                    np.min(feature)
                ])
            
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def process_data(self):
        """Process all audio files and prepare datasets."""
        features = []
        labels = []
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):  # Process only wav files
                    file_path = os.path.join(root, file)
                    genre = os.path.basename(root)  # Genre is the folder name
                    
                    # Extract features from the audio file
                    audio_features = self.extract_features(file_path)
                    
                    if audio_features is not None:
                        features.append(audio_features)
                        labels.append(genre)

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Save the splits
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_test.npy', X_test)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_test.npy', y_test)
        
        # Save label encoder classes
        np.save('data/processed/label_encoder_classes.npy', self.label_encoder.classes_)
        
        return X_train, X_test, y_train, y_test
