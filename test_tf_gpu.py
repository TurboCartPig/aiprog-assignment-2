import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(gpus))
print("Info:\n", gpus)
