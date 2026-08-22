import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from .config import MODEL_PATH

from .dataset import load_dataset


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate():

    # ======================================================
    # LOAD DATA
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
    # LOAD MODEL
    # ======================================================

    print("=" * 60)

    print(
        "LOADING CROP RECOMMENDATION MODEL"
    )

    print("=" * 60)


    model = joblib.load(
        MODEL_PATH
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    y_pred = model.predict(
        X_test
    )


    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    precision = precision_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print("=" * 60)

    print(
        "CROP RECOMMENDATION MODEL EVALUATION"
    )

    print("=" * 60)


    print(

        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"

    )


    print(

        f"Precision : "
        f"{precision * 100:.2f}%"

    )


    print(

        f"Recall    : "
        f"{recall * 100:.2f}%"

    )


    print(

        f"F1 Score  : "
        f"{f1 * 100:.2f}%"

    )


    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("=" * 60)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 60)


    report = classification_report(

        y_test,

        y_pred,

        target_names=label_encoder.classes_,

        zero_division=0

    )


    print(report)


    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(

        y_test,

        y_pred

    )


    print("=" * 60)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 60)


    print(cm)


    # ======================================================
    # PLOT CONFUSION MATRIX
    # ======================================================

    plt.figure(
        figsize=(16, 14)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=label_encoder.classes_,

        yticklabels=label_encoder.classes_

    )


    plt.xlabel(
        "Predicted Crop"
    )


    plt.ylabel(
        "Actual Crop"
    )


    plt.title(
        "Crop Recommendation Confusion Matrix"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.yticks(
        rotation=0
    )


    plt.tight_layout()


    plt.savefig(

        "crop_recommendation_confusion_matrix.png",

        dpi=300

    )


    plt.show()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluate()