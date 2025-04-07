import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EnhancedGenreClassifier:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """Build an enhanced neural network model."""
        model = models.Sequential([
            # Input layer with 56 features
            layers.Dense(128, activation='relu', input_shape=(56,),
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
            
        # Use a more sophisticated optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the model with early stopping and learning rate reduction."""
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_scaled, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

    def save_model(self, model_path='enhanced_model.h5', scaler_path='enhanced_scaler.npy'):
        """Save the trained model and scaler."""
        self.model.save(model_path)
        np.save(scaler_path, [self.scaler.mean_, self.scaler.scale_])

    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @classmethod
    def load_model(cls, model_path='enhanced_model.h5', scaler_path='enhanced_scaler.npy'):
        """Load a trained model and scaler."""
        instance = cls()
        instance.model = models.load_model(model_path)
        scaler_params = np.load(scaler_path, allow_pickle=True)
        instance.scaler.mean_ = scaler_params[0]
        instance.scaler.scale_ = scaler_params[1]
        return instance