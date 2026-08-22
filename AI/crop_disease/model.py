import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.applications import (
    EfficientNetB0,
    MobileNetV2,
)

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from preprocessing import get_data_augmentation

from .config import (
    IMAGE_SIZE,
    MODEL_NAME,
    FREEZE_BASE_MODEL,
    DENSE_UNITS_1,
    DENSE_UNITS_2,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
)


def get_base_model():

    if MODEL_NAME == "EfficientNetB0":

        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
        )

        preprocess = efficientnet_preprocess

    elif MODEL_NAME == "MobileNetV2":

        base_model = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
        )

        preprocess = mobilenet_preprocess

    else:

        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    return base_model, preprocess


def build_model(num_classes):

    inputs = layers.Input(
        shape=(*IMAGE_SIZE, 3),
        name="Input_Image"
    )

    # Data Augmentation
    x = get_data_augmentation()(inputs)

    # Backbone + preprocessing
    base_model, preprocess = get_base_model()

    x = preprocess(x)

    base_model.trainable = not FREEZE_BASE_MODEL

    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        DENSE_UNITS_1,
        activation="relu"
    )(x)

    x = layers.Dropout(DROPOUT_RATE_1)(x)

    x = layers.Dense(
        DENSE_UNITS_2,
        activation="relu"
    )(x)

    x = layers.Dropout(DROPOUT_RATE_2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="CropDiseaseModel"
    )

    return model