import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_data():
    """Split the data into training and testing sets"""
    # Load the features
    df = pd.read_csv('data/features.csv')
    
    # Separate features and labels
    X = df.drop('genre', axis=1).values
    y = df['genre'].values
    
    # Encode the labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    np.save('data/label_encoder_classes.npy', le.classes_)
    
    print("Data split and saved successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

if __name__ == "__main__":
    split_data()