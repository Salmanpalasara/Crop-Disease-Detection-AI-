import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
)


# ==========================================================
# EVALUATE SOIL MODEL
# ==========================================================

def evaluate_model():

    print("=" * 70)
    print("SOIL MODEL EVALUATION")
    print("=" * 70)

    # ======================================================
    # LOAD MODEL
    # ======================================================

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # ======================================================
    # LOAD CLASS NAMES
    # ======================================================

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    print("\nClasses:")

    for index, class_name in enumerate(class_names):

        print(
            f"{index}: {class_name}"
        )

    # ======================================================
    # LOAD TEST DATASET
    # ======================================================

    print("\nLoading test dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(

        TEST_DIR,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False

    )

    print(
        "Test dataset loaded successfully."
    )

    # ======================================================
    # MODEL EVALUATION
    # ======================================================

    print("\nEvaluating model...")

    test_loss, keras_accuracy = model.evaluate(

        test_ds,

        verbose=1

    )

    # ======================================================
    # GET TRUE LABELS
    # ======================================================

    y_true = []

    for images, labels in test_ds:

        y_true.extend(
            labels.numpy()
        )

    y_true = np.array(
        y_true
    )

    # ======================================================
    # GET PREDICTIONS
    # ======================================================

    print("\nGenerating predictions...")

    predictions = model.predict(

        test_ds,

        verbose=1

    )

    y_pred = np.argmax(

        predictions,

        axis=1

    )

    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    precision = precision_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    # ======================================================
    # PRINT METRICS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"Test Loss       : {test_loss:.4f}"
    )

    print(
        f"Keras Accuracy  : {keras_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy        : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision       : {precision * 100:.2f}%"
    )

    print(
        f"Recall          : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score        : {f1 * 100:.2f}%"
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4,

        zero_division=0

    )

    print(report)

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    print("\nClass order:")

    print(
        class_names
    )

    print("\nMatrix:")

    print(cm)

    # ======================================================
    # READABLE CONFUSION MATRIX
    # ======================================================

    print("\n")
    print("=" * 70)
    print("READABLE CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\nRows = Actual Class"
    )

    print(
        "Columns = Predicted Class\n"
    )

    # Header

    print(
        f"{'Actual / Predicted':<20}",
        end=""
    )

    for class_name in class_names:

        print(
            f"{class_name:<18}",
            end=""
        )

    print()

    print("-" * 90)

    for i, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name:<20}",
            end=""
        )

        for j in range(
            len(class_names)
        ):

            print(
                f"{cm[i][j]:<18}",
                end=""
            )

        print()

    # ======================================================
    # COMPLETED
    # ======================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluate_model()