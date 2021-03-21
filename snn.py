"""Task 1.1 STL-10 classification using simple neural network"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras

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
        keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0),
        keras.layers.experimental.preprocessing.Normalization(mean=0.45, variance=0.8),
        keras.layers.Flatten(input_shape=(96, 96, 3)),
        keras.layers.Dense(94, activation="tanh"),
        keras.layers.Dense(10, activation="sigmoid"),
        keras.layers.Softmax(),
    ]
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Fit the model to the training data
model.fit(train, epochs=10)

# Evaluate the performence of the model, based on loss and accuracy
test_loss, test_acc = model.evaluate(test, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
