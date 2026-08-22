import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from .config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR,

        validation_split=0.2,

        subset="training",

        seed=SEED,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="categorical"

    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR,

        validation_split=0.2,

        subset="validation",

        seed=SEED,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="categorical"

    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(

        TEST_DIR,

        shuffle=False,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="categorical"

    )

    return train_dataset, validation_dataset, test_dataset


def get_class_names(dataset):

    return dataset.class_names


def calculate_class_weights(train_dataset):

    labels = []

    for _, y in train_dataset:

        labels.extend(np.argmax(y.numpy(), axis=1))

    labels = np.array(labels)

    classes = np.unique(labels)

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=labels

    )

    class_weights = {

        i: weight

        for i, weight in enumerate(weights)

    }

    return class_weights