import json
import joblib

from sklearn.metrics import accuracy_score

from .config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_NAMES
)

from .dataset import load_dataset

from .model import build_model


# ==========================================================
# TRAIN CROP RECOMMENDATION MODEL
# ==========================================================

def train():

    # ======================================================
    # LOAD DATASET
    # ======================================================

    (

        X_train,

        X_val,

        X_test,

        y_train,

        y_val,

        y_test,

        label_encoder

    ) = load_dataset()


    # ======================================================
    # NUMBER OF CLASSES
    # ======================================================

    num_classes = len(
        label_encoder.classes_
    )


    print("=" * 60)

    print(
        "Number of crop classes:",
        num_classes
    )

    print("=" * 60)


    # ======================================================
    # BUILD MODEL
    # ======================================================

    model = build_model(
        num_classes
    )


    # ======================================================
    # TRAIN MODEL
    # ======================================================

    print("=" * 60)

    print(
        "TRAINING XGBOOST MODEL"
    )

    print("=" * 60)


    model.fit(

        X_train,

        y_train,

        eval_set=[

            (
                X_val,
                y_val
            )

        ],

        verbose=True

    )


    # ======================================================
    # VALIDATION
    # ======================================================

    print("=" * 60)

    print(
        "VALIDATION RESULT"
    )

    print("=" * 60)


    val_predictions = model.predict(
        X_val
    )


    val_accuracy = accuracy_score(

        y_val,

        val_predictions

    )


    print(

        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"

    )


    # ======================================================
    # TEST
    # ======================================================

    print("=" * 60)

    print(
        "TEST RESULT"
    )

    print("=" * 60)


    test_predictions = model.predict(
        X_test
    )


    test_accuracy = accuracy_score(

        y_test,

        test_predictions

    )


    print(

        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"

    )


    # ======================================================
    # SAVE MODEL
    # ======================================================

    joblib.dump(

        model,

        MODEL_PATH

    )


    print(
        "Model saved:",
        MODEL_PATH
    )


    # ======================================================
    # SAVE LABEL ENCODER
    # ======================================================

    joblib.dump(

        label_encoder,

        LABEL_ENCODER_PATH

    )


    print(
        "Label encoder saved:",
        LABEL_ENCODER_PATH
    )


    # ======================================================
    # SAVE FEATURE NAMES
    # ======================================================

    with open(

        FEATURE_NAMES_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            FEATURE_NAMES,

            file,

            indent=4

        )


    print(
        "Feature names saved:",
        FEATURE_NAMES_PATH
    )


    # ======================================================
    # COMPLETED
    # ======================================================

    print("=" * 60)

    print(
        "CROP RECOMMENDATION TRAINING COMPLETED"
    )

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train()