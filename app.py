import streamlit as st
import librosa
import numpy as np
from pathlib import Path
import tensorflow as tf
from src.feature_extraction import extract_features

# Set page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    return tf.keras.models.load_model('data/genre_classifier_model.h5')

@st.cache_resource
def load_label_encoder():
    """Load the label encoder classes"""
    return np.load('data/label_encoder_classes.npy', allow_pickle=True)

def predict_genre(audio_file):
    """
    Predict the genre of an audio file
    """
    # Extract features
    features = extract_features(audio_file)
    
    if features is None:
        return None
    
    # Reshape features for model
    features = features.reshape(1, -1)
    
    # Load model and make prediction
    model = load_model()
    predictions = model.predict(features)
    
    # Get genre labels
    genre_labels = load_label_encoder()
    
    # Get top 3 predictions
    top_3_indices = predictions[0].argsort()[-3:][::-1]
    top_3_genres = genre_labels[top_3_indices]
    # Convert predictions to native Python float
    top_3_probs = [float(p) for p in predictions[0][top_3_indices]]
    
    return list(zip(top_3_genres, top_3_probs))

def main():
    st.title("🎵 Music Genre Classifier")
    st.write("""
    Upload an audio file (WAV format) and the model will predict its genre!
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Add predict button
        if st.button('Predict Genre'):
            with st.spinner('Analyzing audio...'):
                # Save uploaded file temporarily
                temp_path = Path('temp_audio.wav')
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Make prediction
                results = predict_genre(temp_path)
                
                # Remove temporary file
                temp_path.unlink()
                
                if results:
                    st.subheader("Predictions:")
                    
                    # Create columns for visualization
                    cols = st.columns(3)
                    
                    # Display top 3 predictions with progress bars
                    for i, (genre, prob) in enumerate(results):
                        with cols[i]:
                            st.metric(
                                label=f"#{i+1} Prediction",
                                value=genre.title(),
                                delta=f"{prob*100:.1f}%"
                            )
                            st.progress(prob)
                else:
                    st.error("Error processing the audio file. Please try another file.")
    
    # Add information about supported genres
    st.sidebar.title("About")
    st.sidebar.write("""
    This model can classify music into the following genres:
    - Blues
    - Classical
    - Country
    - Disco
    - Hip-Hop
    - Jazz
    - Metal
    - Pop
    - Reggae
    - Rock
    
    For best results:
    - Use WAV format audio files
    - Audio should be at least 30 seconds long
    - Avoid files with multiple genres
    """)

    # Add example section
    st.sidebar.title("Test Examples")
    if st.sidebar.button("Download Test Samples"):
        with st.sidebar.spinner("Downloading..."):
            try:
                from src.download_test_samples import download_test_samples
                download_test_samples()
                st.sidebar.success("Test samples downloaded to 'test_samples' directory!")
            except Exception as e:
                st.sidebar.error(f"Error downloading samples: {str(e)}")

if __name__ == "__main__":
    main()