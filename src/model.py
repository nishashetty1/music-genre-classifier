import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def create_model(input_shape, num_genres):
    """
    Create an improved neural network architecture for music genre classification
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        layers.BatchNormalization(),
        
        # First hidden layer - wider for better feature learning
        layers.Dense(256, kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(128, kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        
        # Third hidden layer
        layers.Dense(64, kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_genres, activation='softmax')
    ])
    
    # Use Adam optimizer with a fixed learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Genre Classification Confusion Matrix')
    plt.ylabel('True Genre')
    plt.xlabel('Predicted Genre')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model():
    """Train the model with improved training process"""
    # Load data
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of genres: {len(np.unique(y_train))}")
    
    # Create model
    model = create_model((X_train.shape[1],), len(np.unique(y_train)))
    model.summary()
    
    # Callbacks
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            'data/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Calculate class weights for imbalanced data
    class_weights = dict(enumerate(
        np.bincount(y_train).max() / np.bincount(y_train)
    ))
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Load the best model
    model = tf.keras.models.load_model('data/best_model.h5')
    
    # Evaluate on test set
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Load genre labels
    genre_labels = np.load('data/label_encoder_classes.npy', allow_pickle=True)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=genre_labels))
    
    # Save confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_classes, genre_labels, 
        'data/confusion_matrix.png'
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/training_history.png')
    
    # Save metadata
    metadata = {
        'training_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'num_features': X_train.shape[1],
        'num_genres': len(genre_labels),
        'test_accuracy': float(test_accuracy),
        'feature_names': list(genre_labels)
    }
    
    np.save('data/model_metadata.npy', metadata)
    print("\nModel and metadata saved successfully!")

if __name__ == "__main__":
    train_model()