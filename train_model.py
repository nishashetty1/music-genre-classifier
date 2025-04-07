import numpy as np
from src.models.genre_classifier import GenreClassifier

# Load the preprocessed data
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')
genre_classes = np.load('data/processed/label_encoder_classes.npy')

# Initialize and train the model
classifier = GenreClassifier(num_classes=len(genre_classes))
history = classifier.train(X_train, y_train, X_test, y_test, epochs=50)

# Plot and save training history
classifier.plot_training_history(history)

# Save the trained model and scaler
classifier.save_model()

# Print final evaluation metrics
test_loss, test_accuracy = classifier.model.evaluate(
    classifier.scaler.transform(X_test), y_test, verbose=0
)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")