import tensorflow as tf

from dataset import load_datasets, calculate_class_weights
from callbacks import get_callbacks

from config import (
    CHECKPOINT_PATH,
    FINAL_MODEL_PATH,
    FINE_TUNE_LEARNING_RATE,
    EPOCHS,
    FINE_TUNE_AT,
    METRICS,
)

print("=" * 60)
print("Fine Tuning Model")
print("=" * 60)

print("\nLoading Dataset...")

train_ds, val_ds, test_ds = load_datasets()

class_weights = calculate_class_weights(train_ds)

print("\nLoading Best Model...")

model = tf.keras.models.load_model(CHECKPOINT_PATH)

# Find the EfficientNet base model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise ValueError("Base model not found!")

print("\nUnfreezing Base Model...")

base_model.trainable = True

# Freeze all layers except the last FINE_TUNE_AT layers
for layer in base_model.layers[:-FINE_TUNE_AT]:
    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),

    loss="categorical_crossentropy",

    metrics=METRICS

)

print("\nStarting Fine Tuning...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=get_callbacks(),

    verbose=1

)

print("\nSaving Final Fine-Tuned Model...")

model.save(FINAL_MODEL_PATH)

print("\nFine Tuning Completed Successfully.")