import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# ==========================================================
# Data Augmentation
# ==========================================================

def get_data_augmentation():

    return tf.keras.Sequential([

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),

    ], name="soil_data_augmentation")


# ==========================================================
# Build Model
# ==========================================================

def build_model(
    num_classes
):

    # ------------------------------------------------------
    # Data Augmentation
    # ------------------------------------------------------

    augmentation = (
        get_data_augmentation()
    )


    # ------------------------------------------------------
    # Pretrained EfficientNetB0
    # ------------------------------------------------------

    base_model = EfficientNetB0(

        include_top=False,

        weights="imagenet",

        input_shape=(
            224,
            224,
            3
        )

    )


    # Freeze pretrained layers

    base_model.trainable = False


    # ------------------------------------------------------
    # Input
    # ------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            224,
            224,
            3
        )
    )


    # ------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------

    x = augmentation(
        inputs
    )


    # ------------------------------------------------------
    # EfficientNet
    # ------------------------------------------------------

    x = base_model(
        x,
        training=False
    )


    # ------------------------------------------------------
    # Global Pooling
    # ------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(
        x
    )


    # ------------------------------------------------------
    # Batch Normalization
    # ------------------------------------------------------

    x = layers.BatchNormalization()(
        x
    )


    # ------------------------------------------------------
    # Dense
    # ------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu"
    )(x)


    # ------------------------------------------------------
    # Dropout
    # ------------------------------------------------------

    x = layers.Dropout(
        0.4
    )(x)


    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    outputs = layers.Dense(

        num_classes,

        activation="softmax"

    )(x)


    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs

    )


    return model, base_model