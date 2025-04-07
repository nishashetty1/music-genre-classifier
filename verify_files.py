import os

def verify_files():
    required_files = [
        'models/enhanced_model.h5',
        'data/processed/enhanced_scaler.npy',
        'data/processed/enhanced_label_encoder_classes.npy',
        'src/app/streamlit_app.py',
        'src/models/enhanced_genre_classifier.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"- {file}")
    else:
        print("All required files are present!")

if __name__ == "__main__":
    verify_files()