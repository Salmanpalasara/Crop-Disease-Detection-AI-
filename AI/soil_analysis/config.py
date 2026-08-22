from pathlib import Path


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# print("PROJECT ROOT:", PROJECT_ROOT)
# print("DATASET DIR:", PROJECT_ROOT / "Dataset" / "Soil_dataset")
# print("TRAIN DIR:", PROJECT_ROOT / "Dataset" / "Soil_dataset" / "train")
# print(
#     "TRAIN EXISTS:",
#     (PROJECT_ROOT / "Dataset" / "Soil_dataset" / "train").exists()
# )
# print(
#     "TEST EXISTS:",
#     (PROJECT_ROOT / "Dataset" / "Soil_dataset" / "test").exists()
# )
# print("=" * 60)

# ==========================================================
# DATASET
# ==========================================================


DATASET_DIR = (
    PROJECT_ROOT.parent
    / "Dataset"
    / "Soil_dataset"
)

TRAIN_DIR = DATASET_DIR / "train"

TEST_DIR = DATASET_DIR / "test"
# ==========================================================
# MODEL DIRECTORY
# ==========================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "AI"
    / "soil_analysis"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = (
    MODEL_DIR
    / "soil_model.keras"
)


CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)


# ==========================================================
# IMAGE
# ==========================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)

CHANNELS = 3


# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 16

EPOCHS = 25

LEARNING_RATE = 1e-3

FINE_TUNE_LEARNING_RATE = 1e-5