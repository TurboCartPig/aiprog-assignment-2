"""Task 1.1 STL-10 classification using simple neural network"""

import tensorflow as tf
import tensorflow.keras as ka
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

train, test = tfds.load(
    "stl10",
    split=["train", "test"],
    as_supervised=True,
    shuffle_files=False,
    batch_size=64,
    download=False,
    data_dir="./datasets/",
)

model = ka.Sequential(
    [
        ka.layers.experimental.preprocessing.Rescaling(1.0 / 255.0),
        ka.layers.experimental.preprocessing.Normalization(mean=0.45, variance=0.8),
        ka.layers.Flatten(input_shape=(96, 96, 3)),
        ka.layers.Dense(94, activation="tanh"),
        ka.layers.Dense(10, activation="sigmoid"),
        ka.layers.Softmax(),
    ]
)

model.compile(
    optimizer=ka.optimizers.Adam(),
    loss=ka.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train, epochs=10)

test_loss, test_acc = model.evaluate(test, verbose=2)

print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
