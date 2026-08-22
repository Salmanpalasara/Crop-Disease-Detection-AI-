import json

import tensorflow as tf

from .config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    EPOCHS,
    LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
)

from .dataset import load_datasets

from .model import build_model


# ==========================================================
# Train Soil Model
# ==========================================================

def train():

    # ======================================================
    # LOAD DATASET
    # ======================================================

    (
        train_ds,
        val_ds,
        test_ds,
        class_names
    ) = load_datasets()


    num_classes = len(
        class_names
    )


    print(
        "Number of classes:",
        num_classes
    )


    # ======================================================
    # BUILD MODEL
    # ======================================================

    model, base_model = build_model(

        num_classes

    )


    # ======================================================
    # MODEL SUMMARY
    # ======================================================

    model.summary()


    # ======================================================
    # CALLBACKS
    # ======================================================

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.2,

            patience=2,

            min_lr=1e-7,

            verbose=1

        ),

        tf.keras.callbacks.ModelCheckpoint(

            filepath=MODEL_PATH,

            monitor="val_accuracy",

            save_best_only=True,

            verbose=1

        )

    ]


    # ======================================================
    # PHASE 1
    # TRANSFER LEARNING
    # ======================================================

    print("=" * 60)

    print(
        "PHASE 1: TRANSFER LEARNING"
    )

    print("=" * 60)


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks

    )


    # ======================================================
    # PHASE 2
    # FINE TUNING
    # ======================================================

    print("=" * 60)

    print(
        "PHASE 2: FINE TUNING"
    )

    print("=" * 60)


    # ------------------------------------------------------
    # Unfreeze EfficientNet
    # ------------------------------------------------------

    base_model.trainable = True


    # ------------------------------------------------------
    # Freeze Earlier Layers
    # ------------------------------------------------------

    fine_tune_from = 180


    for layer in base_model.layers[:

        fine_tune_from

    ]:

        layer.trainable = False


    # ------------------------------------------------------
    # Keep BatchNormalization Frozen
    # ------------------------------------------------------

    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False


    # ======================================================
    # RECOMPILE
    # ======================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=FINE_TUNE_LEARNING_RATE

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    # ======================================================
    # FINE TUNE
    # ======================================================

    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


    # ======================================================
    # LOAD BEST MODEL
    # ======================================================

    print("=" * 60)

    print(
        "Loading best model..."
    )

    print("=" * 60)


    best_model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    best_model.save(
        MODEL_PATH
    )


    # ======================================================
    # SAVE CLASS NAMES
    # ======================================================

    with open(

        CLASS_NAMES_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            class_names,

            file,

            ensure_ascii=False,

            indent=4

        )


    # ======================================================
    # FINAL TEST EVALUATION
    # ======================================================

    print("=" * 60)

    print(
        "FINAL TEST EVALUATION"
    )

    print("=" * 60)


    test_loss, test_accuracy = (
        best_model.evaluate(
            test_ds,
            verbose=1
        )
    )


    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


    # ======================================================
    # COMPLETED
    # ======================================================

    print("=" * 60)

    print(
        "SOIL MODEL TRAINING COMPLETED"
    )

    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Classes:",
        class_names
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train()