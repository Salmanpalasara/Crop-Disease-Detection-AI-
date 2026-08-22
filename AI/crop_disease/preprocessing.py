import tensorflow as tf

from config import (
    RANDOM_FLIP,
    ROTATION_FACTOR,
    ZOOM_FACTOR,
    CONTRAST_FACTOR,
    BRIGHTNESS_FACTOR,
    TRANSLATION_HEIGHT,
    TRANSLATION_WIDTH,
)


def get_data_augmentation():
    """
    Returns the data augmentation pipeline.

    These augmentations are applied only during training
    to improve model generalization and reduce overfitting.
    """

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                RANDOM_FLIP,
                name="RandomFlip"
            ),

            tf.keras.layers.RandomRotation(
                ROTATION_FACTOR,
                name="RandomRotation"
            ),

            tf.keras.layers.RandomZoom(
                height_factor=ZOOM_FACTOR,
                width_factor=ZOOM_FACTOR,
                name="RandomZoom"
            ),

            tf.keras.layers.RandomContrast(
                CONTRAST_FACTOR,
                name="RandomContrast"
            ),

            tf.keras.layers.RandomBrightness(
                factor=BRIGHTNESS_FACTOR,
                value_range=(0, 255),
                name="RandomBrightness"
            ),

            tf.keras.layers.RandomTranslation(
                height_factor=TRANSLATION_HEIGHT,
                width_factor=TRANSLATION_WIDTH,
                name="RandomTranslation"
            ),
        ],
        name="DataAugmentation",
    )

    return augmentation