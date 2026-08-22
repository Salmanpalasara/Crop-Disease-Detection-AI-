import tensorflow as tf


def get_data_augmentation():

    return tf.keras.Sequential([

        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.15),

        tf.keras.layers.RandomZoom(0.15),

        tf.keras.layers.RandomContrast(0.10),

    ])