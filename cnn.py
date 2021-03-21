"""Task 1.2 STL-10 classification using convolutional neural network"""

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Load the dataset split into training data (images and labels), and testing data.
train, test = tfds.load(
    "stl10",
    split=["train", "test"],
    as_supervised=True,
    shuffle_files=False,
    batch_size=64,
    download=False,
    data_dir="./datasets/",
)

# Define the model
model = keras.Sequential(
    [
        preprocessing.Rescaling(1.0 / 255.0),
        preprocessing.Normalization(mean=0.45, variance=0.8),
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=(96, 96, 3)),
        layers.MaxPooling2D(3, 3),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="sigmoid"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Fit the model to the training data
model.fit(train, validation_data=test, epochs=7)

# Evaluate the performance of the model, based on loss and accuracy
test_loss, test_acc = model.evaluate(test, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
