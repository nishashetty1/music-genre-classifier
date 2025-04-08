import streamlit as st
import numpy as np
import librosa
import os

# Add this at the top to set up proper paths
import sys
from pathlib import Path

# Get the absolute path of the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from src.models.enhanced_genre_classifier import EnhancedGenreClassifier

# Set page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1db954;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the enhanced model and genre classes
@st.cache_resource
def load_model():
    try:
        # Use ROOT_DIR to construct absolute paths
        model_path = os.path.join(ROOT_DIR, 'models', 'enhanced_model.h5')
        scaler_path = os.path.join(ROOT_DIR, 'data', 'processed', 'enhanced_scaler.npy')
        classes_path = os.path.join(ROOT_DIR, 'data', 'processed', 'enhanced_label_encoder_classes.npy')
        
        # Load the model
        classifier = EnhancedGenreClassifier.load_model(
            model_path=model_path,
            scaler_path=scaler_path
        )
        
        # Load genre classes
        genre_classes = np.load(classes_path)
        
        print(f"Model loaded successfully from: {model_path}")
        print(f"Scaler loaded successfully from: {scaler_path}")
        print(f"Classes loaded successfully from: {classes_path}")
        
        return classifier, genre_classes
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Detailed error: {str(e)}")  # More detailed error in console
        return None, None

# Rest of your existing feature extraction code...
def extract_features(audio_file, sr=22050, duration=30):
    """Extract features using the enhanced feature set."""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_file, duration=duration, sr=sr)
        
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
        
        # Verify feature length
        if len(features) != 56:
            raise ValueError(f"Expected 56 features but got {len(features)}")
            
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        print(f"Detailed error in feature extraction: {str(e)}")  # More detailed error in console
        return None

def main():
    st.title('🎵 Enhanced Music Genre Classification')
    st.write('Upload an audio file (.wav format) to classify its genre')

    # Add sidebar with information
    st.sidebar.title("About")
    st.sidebar.info(
        "This web-app uses a neural network model to classify music into different genres. "
        "The model achieves 73.50% accuracy on the test set.\n\n"
        "Supported genres:\n"
        "- Blues\n"
        "- Classical\n"
        "- Country\n"
        "- Disco\n"
        "- Hip Hop\n"
        "- Jazz\n"
        "- Metal\n"
        "- Pop\n"
        "- Reggae\n"
        "- Rock"
    )

    # Load the model and genre classes
    classifier, genre_classes = load_model()

    if classifier is None or genre_classes is None:
        st.error("Failed to load the model. Please check the console for detailed error messages.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=['wav'])

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Display audio player
            st.audio(uploaded_file)
            if st.button('Classify', type='primary'):
                with st.spinner('Analyzing audio...'):
                    # Extract features
                    features = extract_features(uploaded_file)
                    
                    if features is not None:
                        try:
                            # Make prediction
                            predictions = classifier.predict(features)[0]
                            
                            # Get top 3 predictions
                            top3_idx = predictions.argsort()[-3:][::-1]
                            
                            with col2:
                                st.write("### Classification Results")
                                
                                # Create a container for results
                                results_container = st.container()
                                
                                with results_container:
                                    # Display top 3 predictions with confidence bars
                                    for idx in top3_idx:
                                        confidence = float(predictions[idx] * 100)
                                        genre = str(genre_classes[idx])
                                        st.write(f"**{genre.title()}**: {confidence:.1f}%")
                                        st.progress(float(predictions[idx]))
                                    
                                    # Show distribution plot
                                    st.write("### Confidence Distribution")
                                    import matplotlib.pyplot as plt
                                    
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    genres = [str(genre_classes[i]) for i in range(len(genre_classes))]
                                    confidences = [float(pred * 100) for pred in predictions]
                                    
                                    ax.bar(genres, confidences)
                                    plt.xticks(rotation=45, ha='right')
                                    plt.ylabel('Confidence (%)')
                                    plt.title('Genre Prediction Distribution')
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    st.info(
                                        "💡 The confidence scores indicate how sure the model is "
                                        "about its prediction. Higher percentages mean more confidence."
                                    )
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            print(f"Detailed prediction error: {str(e)}")  # More detailed error in console
    else:
        st.info("👆 Please upload a .wav file to get started.")

    # Add footer
    st.markdown("---")

if __name__ == "__main__":
    main()