# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Step 1: Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten images into 1D arrays
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

# Step 2: Define the neural network architecture
model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(784,)),  # Hidden Layer
        Dense(10, activation="softmax"),  # Output Layer
    ]
)

# Step 3: Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Step 4: Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Step 6: Save the trained model (optional)
model.save("mnist_classifier.h5")
print("Model saved as 'mnist_classifier.h5'")
