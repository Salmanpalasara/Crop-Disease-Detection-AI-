import tensorflow as tf

from .config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
)


# ==========================================================
# Load Datasets
# ==========================================================

def load_datasets():

    # ======================================================
    # LOAD FULL TRAIN DATASET
    # ======================================================

    full_train_ds = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=True,

        seed=42

    )


    # ======================================================
    # CLASS NAMES
    # ======================================================

    class_names = full_train_ds.class_names


    print("=" * 60)

    print("SOIL DATASET")

    print("=" * 60)

    print("Classes:")

    for index, class_name in enumerate(class_names):

        print(
            f"{index}: {class_name}"
        )

    print("=" * 60)


    # ======================================================
    # SPLIT TRAIN DATA
    # ======================================================

    total_batches = tf.data.experimental.cardinality(
        full_train_ds
    ).numpy()


    train_batches = int(
        total_batches * 0.8
    )

    val_batches = total_batches - train_batches


    # ======================================================
    # TRAIN DATA
    # ======================================================

    train_ds = full_train_ds.take(
        train_batches
    )


    # ======================================================
    # VALIDATION DATA
    # ======================================================

    val_ds = full_train_ds.skip(
        train_batches
    )


    # ======================================================
    # TEST DATASET
    # ======================================================

    test_ds = tf.keras.utils.image_dataset_from_directory(

        TEST_DIR,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False

    )


    # ======================================================
    # PERFORMANCE OPTIMIZATION
    # ======================================================

    AUTOTUNE = tf.data.AUTOTUNE


    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        AUTOTUNE
    )


    # ======================================================
    # RETURN
    # ======================================================

    return (

        train_ds,

        val_ds,

        test_ds,

        class_names

    )