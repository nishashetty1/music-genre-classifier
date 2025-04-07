# Music Genre Classifier

A neural network model that classifies music into different genres using audio features.

## Setup

1. Install dependencies:
```pip install -r requirements.txt```

2. Run the Streamlit app:
```streamlit run src/app/streamlit_app.py```

## Features

- Classifies music into 10 different genres
- Uses advanced audio feature extraction
- Real-time classification through web interface
- Supports WAV audio files

## Model Architecture

- Enhanced neural network with multiple layers
- Uses MFCCs, spectral features, and chroma features
- Achieves 73.50% accuracy on test set