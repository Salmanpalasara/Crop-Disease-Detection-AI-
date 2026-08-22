import time

import pandas as pd
import tensorflow as tf

from dataset import load_datasets, calculate_class_weights
from model import build_model
from callbacks import get_callbacks

from config import (
    LEARNING_RATE,
    EPOCHS,
    METRICS,
    FINAL_MODEL_PATH,
    HISTORY_PATH,
)

print("=" * 60)
print("AI Powered Crop Disease Detection")
print("=" * 60)

start_time = time.time()

# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading Dataset...\n")

train_ds, val_ds, test_ds = load_datasets()

num_classes = len(train_ds.class_names)

print(f"Total Classes : {num_classes}")

# ==========================================================
# Class Weights
# ==========================================================

print("\nCalculating Class Weights...")

class_weights = calculate_class_weights(train_ds)

print("Done")

# ==========================================================
# Build Model
# ==========================================================

print("\nBuilding Model...\n")

model = build_model(num_classes)

# ==========================================================
# Compile Model
# ==========================================================

print("Compiling Model...\n")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="categorical_crossentropy",

    metrics=METRICS

)

print("Model Summary\n")

model.summary()

# ==========================================================
# Train
# ==========================================================

print("\nStarting Training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=get_callbacks(),

    verbose=1

)

# ==========================================================
# Save Final Model
# ==========================================================

print("\nSaving Model...\n")

model.save(FINAL_MODEL_PATH)

print("Model Saved Successfully.")

# ==========================================================
# Save History
# ==========================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(HISTORY_PATH, index=False)

print("Training History Saved.")

# ==========================================================
# Evaluate
# ==========================================================

print("\nEvaluating Model...\n")

test_loss, *metrics = model.evaluate(test_ds, verbose=1)

print("\nTest Loss :", round(test_loss, 4))

metric_names = model.metrics_names[1:]

for name, value in zip(metric_names, metrics):
    print(f"{name} : {value:.4f}")

# ==========================================================
# Training Time
# ==========================================================

end_time = time.time()

training_time = (end_time - start_time) / 60

print("\nTraining Completed Successfully.")

print(f"Training Time : {training_time:.2f} Minutes")

print(f"Final Model : {FINAL_MODEL_PATH}")

print(f"History : {HISTORY_PATH}")

print("=" * 60)