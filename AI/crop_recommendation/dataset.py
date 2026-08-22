import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    DATASET_PATH,
    FEATURE_NAMES,
    TARGET_COLUMN,
    RANDOM_STATE
)


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    print("=" * 60)
    print("CROP RECOMMENDATION DATASET")
    print("=" * 60)

    print(
        "Dataset:",
        DATASET_PATH
    )


    # ------------------------------------------------------
    # Check dataset
    # ------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(

            f"Dataset not found:\n{DATASET_PATH}"

        )


    # ------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )


    print(
        "Dataset Shape:",
        df.shape
    )


    print(
        "Columns:",
        list(df.columns)
    )


    # ======================================================
    # CHECK REQUIRED COLUMNS
    # ======================================================

    required_columns = (
        FEATURE_NAMES
        + [TARGET_COLUMN]
    )


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        raise ValueError(

            "Missing columns: "
            + str(missing_columns)

        )


    # ======================================================
    # REMOVE MISSING VALUES
    # ======================================================

    print(
        "Missing values before cleaning:"
    )

    print(
        df[required_columns].isnull().sum()
    )


    df = df.dropna(
        subset=required_columns
    )


    # ======================================================
    # FEATURES
    # ======================================================

    X = df[
        FEATURE_NAMES
    ].copy()


    # ======================================================
    # TARGET
    # ======================================================

    y = df[
        TARGET_COLUMN
    ].astype(str)


    # ======================================================
    # LABEL ENCODING
    # ======================================================

    label_encoder = LabelEncoder()


    y_encoded = label_encoder.fit_transform(
        y
    )


    # ======================================================
    # PRINT CLASSES
    # ======================================================

    print("=" * 60)

    print(
        "CROP CLASSES"
    )

    print("=" * 60)


    for index, crop in enumerate(
        label_encoder.classes_
    ):

        print(
            f"{index}: {crop}"
        )


    print(
        "Total Classes:",
        len(label_encoder.classes_)
    )


    # ======================================================
    # TRAIN / TEMP SPLIT
    # ======================================================

    X_train, X_temp, y_train, y_temp = (
        train_test_split(

            X,

            y_encoded,

            test_size=0.20,

            random_state=RANDOM_STATE,

            stratify=y_encoded

        )
    )


    # ======================================================
    # VALIDATION / TEST SPLIT
    # ======================================================

    X_val, X_test, y_val, y_test = (
        train_test_split(

            X_temp,

            y_temp,

            test_size=0.50,

            random_state=RANDOM_STATE,

            stratify=y_temp

        )
    )


    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    print("=" * 60)

    print(
        "DATA SPLIT"
    )

    print("=" * 60)


    print(
        "Total:",
        len(X)
    )

    print(
        "Training:",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_val)
    )

    print(
        "Testing:",
        len(X_test)
    )


    return (

        X_train,

        X_val,

        X_test,

        y_train,

        y_val,

        y_test,

        label_encoder

    )