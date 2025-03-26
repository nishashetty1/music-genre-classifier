import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_features(audio_path):
    """
    Extract a stable set of audio features for genre classification
    """
    try:
        # Load the audio file with a fixed duration and sample rate
        y, sr = librosa.load(audio_path, duration=30, sr=22050)
        
        # Initialize feature list
        features = []
        
        # 1. MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        features.extend(mfcc_means)
        features.extend(mfcc_stds)
        
        # 2. Spectral Features
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # 3. Zero Crossing Rate
        zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zero_crossings), np.std(zero_crossings)])
        
        # 4. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Convert to numpy array and ensure all values are finite
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def process_dataset():
    """
    Process the GTZAN dataset with stable feature extraction
    """
    data_dir = Path('data/Data/genres_original')
    features_list = []
    labels = []
    failed_files = []
    
    for genre_folder in data_dir.iterdir():
        if genre_folder.is_dir():
            genre = genre_folder.name
            print(f"Processing {genre} files...")
            
            for audio_file in tqdm(list(genre_folder.glob('*.wav'))):
                feature_vector = extract_features(audio_file)
                if feature_vector is not None:
                    features_list.append(feature_vector)
                    labels.append(genre)
                else:
                    failed_files.append(str(audio_file))
    
    if failed_files:
        print("\nFailed to process the following files:")
        for file in failed_files:
            print(f"- {file}")
    
    # Convert to numpy array
    features_array = np.array(features_list)
    
    # Create feature names
    feature_names = (
        # MFCC features
        [f'mfcc{i}_mean' for i in range(13)] +
        [f'mfcc{i}_std' for i in range(13)] +
        # Spectral features
        ['spectral_centroid_mean', 'spectral_centroid_std',
         'spectral_rolloff_mean', 'spectral_rolloff_std'] +
        # Zero crossing rate
        ['zero_crossing_mean', 'zero_crossing_std'] +
        # RMS energy
        ['rms_mean', 'rms_std']
    )
    
    # Create DataFrame
    df = pd.DataFrame(features_array, columns=feature_names)
    df['genre'] = labels
    
    # Save features
    output_path = Path('data/features.csv')
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to {output_path}")
    print(f"Total samples processed: {len(features_list)}")
    print(f"Number of features per sample: {len(feature_names)}")
    
    # Print basic statistics
    print("\nFeature Statistics Summary:")
    print(df.describe().round(2))
    
    return df

if __name__ == "__main__":
    process_dataset()