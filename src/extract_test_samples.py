import numpy as np
import shutil
from pathlib import Path

def extract_test_samples():
    """
    Extract some test samples from our test set
    """
    # Load test data indices
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    genre_labels = np.load('data/label_encoder_classes.npy', allow_pickle=True)
    
    # Create test_samples directory
    test_dir = Path('test_samples')
    test_dir.mkdir(exist_ok=True)
    
    # Source directory of the GTZAN dataset
    data_dir = Path('data/Data/genres_original')
    
    # Copy a few samples from each genre
    for genre in genre_labels:
        genre_dir = data_dir / genre
        if genre_dir.exists():
            # Get first test file from each genre
            source_file = next(genre_dir.glob('*.wav'))
            dest_file = test_dir / f'test_{genre}.wav'
            
            # Copy the file
            shutil.copy2(source_file, dest_file)
            print(f"Copied test file for genre {genre}")

if __name__ == "__main__":
    extract_test_samples()