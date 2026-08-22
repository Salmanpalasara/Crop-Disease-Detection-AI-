import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from dataset import load_datasets

from config import (
    FINAL_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH,
    METRICS_PATH,
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

_, _, test_ds = load_datasets()

class_names = test_ds.class_names

# ==========================================================
# Load Model
# ==========================================================

print("\nLoading Trained Model...\n")

model = tf.keras.models.load_model(FINAL_MODEL_PATH)

# ==========================================================
# Prediction
# ==========================================================

print("Predicting Test Dataset...\n")

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

y_true = np.argmax(y_true, axis=1)

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

print("=" * 60)
print("Evaluation Metrics")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ==========================================================
# Save Metrics
# ==========================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(METRICS_PATH, index=False)

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report\n")
print(report)

with open(CLASSIFICATION_REPORT_PATH, "w") as file:
    file.write(report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(20, 18))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True
)

plt.title("Crop Disease Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)

plt.show()

print("\nConfusion Matrix Saved.")

print("Classification Report Saved.")

print("Evaluation Metrics Saved.")

print("\nEvaluation Completed Successfully.")