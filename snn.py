"""Task 1.1 STL-10 classification using simple neural network"""

import tensorflow as tf
import tensorflow_datasets as tfds

images, lables = tfds.load(
    "stl10",
    split="train",
    as_supervised=True,
    download=True,
    data_dir="./datasets/",
)
