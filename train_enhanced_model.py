import numpy as np
import librosa
from src.models.enhanced_genre_classifier import EnhancedGenreClassifier
import os

def extract_enhanced_features(file_path, sr=22050, duration=30):
    """Extract an enhanced set of features from audio file."""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, duration=duration, sr=sr)
        
        features = []
        
        # 1. MFCCs (13 coefficients × 2 statistics = 26 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for mfcc in mfccs:
            features.extend([np.mean(mfcc), np.std(mfcc)])
            
        # 2. Spectral Centroid (2 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        
        # 3. Spectral Rolloff (2 features)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.extend([np.mean(rolloff), np.std(rolloff)])
        
        # 4. Zero Crossing Rate (2 features)
        zero_crossing = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zero_crossing), np.std(zero_crossing)])
        
        # 5. Chroma Features (12 features × 2 statistics = 24 features)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        for chroma_band in chroma:
            features.extend([np.mean(chroma_band), np.std(chroma_band)])
            
        # Total features so far: 26 + 2 + 2 + 2 + 24 = 56 features
        
        # Verify feature length
        if len(features) != 56:
            print(f"Warning: Got {len(features)} features instead of 56")
            return None
            
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_dataset(data_path):
    """Process all audio files with enhanced feature extraction."""
    features = []
    labels = []
    genres = sorted(os.listdir(data_path))
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue
            
        print(f"\nProcessing {genre} files...")
        files_processed = 0
        
        for file in sorted(os.listdir(genre_path)):
            if not file.endswith('.wav'):
                continue
                
            file_path = os.path.join(genre_path, file)
            print(f"Processing {file_path}")
            
            # Extract enhanced features
            audio_features = extract_enhanced_features(file_path)
            
            if audio_features is not None:
                features.append(audio_features)
                labels.append(genre)
                files_processed += 1
                
        print(f"Successfully processed {files_processed} files for genre {genre}")
    
    if not features:
        raise ValueError("No features were successfully extracted!")
    
    # Convert to numpy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    
    print(f"\nFinal dataset shape:")
    print(f"Features matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

if __name__ == "__main__":
    # Create processed data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Process the dataset with enhanced features
        print("Extracting enhanced features...")
        X, y = process_dataset('data/genres_original')
        
        # Convert labels to numerical values
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split the data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Save the label encoder classes
        np.save('data/processed/enhanced_label_encoder_classes.npy', label_encoder.classes_)
        
        # Initialize and train the enhanced model
        print("\nTraining enhanced model...")
        classifier = EnhancedGenreClassifier(num_classes=len(label_encoder.classes_))
        
        # Train with smaller batch size and show progress bar
        history = classifier.train(
            X_train, y_train, 
            X_test, y_test, 
            epochs=100,
            batch_size=16  # Reduced batch size for better stability
        )
        
        # Save the enhanced model
        classifier.save_model()
        
        # Print final evaluation metrics
        test_loss, test_accuracy = classifier.model.evaluate(
            classifier.scaler.transform(X_test), y_test, verbose=1
        )
        print(f"\nEnhanced Model Performance:")
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        if 'classifier' in locals() and hasattr(classifier, 'model'):
            classifier.save_model(model_path='interrupted_model.h5')
            print("Model saved as 'interrupted_model.h5'")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")