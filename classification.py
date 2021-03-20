"""Training exercies before assignment 2"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"Tensorflow version: {tf.__version__}")

# Define and load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Show one of the images from the dataset
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Scale the images to 0..1 scale
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show a collection of images and their labels
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])

# plt.show()

# Define the neural net model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

# Set additional settings for the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Fit the model to the training data
model.fit(train_images, train_labels, epochs=10)

# Evaluate the models performace based on accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest loss: {test_loss}")
print(f"Test accuracy: {test_acc}")
