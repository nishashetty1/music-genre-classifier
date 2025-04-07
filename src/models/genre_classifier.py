import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class GenreClassifier:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """Build and compile the neural network model."""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(12,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the model and return training history."""
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_scaled, y_test),
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training & validation accuracy and loss."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def save_model(self, model_path='model.h5', scaler_path='scaler.npy'):
        """Save the trained model and scaler."""
        self.model.save(model_path)
        np.save(scaler_path, [self.scaler.mean_, self.scaler.scale_])
    
    @classmethod
    def load_model(cls, model_path='model.h5', scaler_path='scaler.npy'):
        """Load a trained model and scaler."""
        instance = cls()
        instance.model = models.load_model(model_path)
        scaler_params = np.load(scaler_path, allow_pickle=True)
        instance.scaler.mean_ = scaler_params[0]
        instance.scaler.scale_ = scaler_params[1]
        return instance
