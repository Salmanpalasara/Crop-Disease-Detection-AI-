from pathlib import Path
import tensorflow as tf

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "Dataset"

TRAIN_DIR = DATASET_DIR / "Crop_dataset" / "train"
TEST_DIR = DATASET_DIR / "Crop_dataset" / "test"

MODEL_DIR = PROJECT_ROOT / "capstone" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "crop_model.keras"

LOG_DIR = MODEL_DIR / "logs"

# ==========================================================
# Data Augmentation
# ==========================================================

RANDOM_FLIP = "horizontal"

ROTATION_FACTOR = 0.15

ZOOM_FACTOR = 0.15

CONTRAST_FACTOR = 0.10

BRIGHTNESS_FACTOR = 0.10

TRANSLATION_HEIGHT = 0.10

TRANSLATION_WIDTH = 0.10


# ==========================================================
# Image Configuration
# ==========================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3

IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 8

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# Training
# ==========================================================

EPOCHS = 25

LEARNING_RATE = 1e-3

FINE_TUNE_LEARNING_RATE = 1e-5

FINE_TUNE_AT = 30

# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "EfficientNetB0"

FREEZE_BASE_MODEL = True

DENSE_UNITS_1 = 512
DENSE_UNITS_2 = 256

DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.3

# ==========================================================
# Callbacks
# ==========================================================

PATIENCE = 5

REDUCE_LR_PATIENCE = 2

MIN_LR = 1e-6

# ==========================================================
# Training Files
# ==========================================================

CHECKPOINT_PATH = MODEL_DIR / "best_crop_model.keras"

FINAL_MODEL_PATH = MODEL_DIR / "crop_model_final.keras"

HISTORY_PATH = MODEL_DIR / "training_history.csv"

# ==========================================================
# Metrics
# ==========================================================

METRICS = [

    "accuracy",

    tf.keras.metrics.Precision(name="precision"),

    tf.keras.metrics.Recall(name="recall"),

]

# ==========================================================
# Evaluation Files
# ==========================================================

CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

CLASSIFICATION_REPORT_PATH = MODEL_DIR / "classification_report.txt"

METRICS_PATH = MODEL_DIR / "evaluation_metrics.csv"

# ==========================================================
# History
# ==========================================================

HISTORY_PATH = MODEL_DIR / "training_history.csv"
# ==========================================================
# Random Seed
# ==========================================================

SEED = 42