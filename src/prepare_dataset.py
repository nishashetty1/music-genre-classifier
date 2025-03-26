import os
import zipfile
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_gtzan_dataset():
    """
    Download and extract the GTZAN dataset from Kaggle
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    print("Downloading GTZAN dataset from Kaggle...")
    api.dataset_download_files('andradaolteanu/gtzan-dataset-music-genre-classification',
                             path='data',
                             unzip=True)
    
    print("Dataset downloaded and extracted successfully!")
    
    # Clean up zip file if it exists
    zip_path = data_dir / 'gtzan-dataset-music-genre-classification.zip'
    if zip_path.exists():
        os.remove(zip_path)

if __name__ == "__main__":
    download_gtzan_dataset()