import json
import joblib
import numpy as np


from .config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    LABEL_ENCODER_PATH,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(
    MODEL_PATH
)


# ==========================================================
# LOAD FEATURE NAMES
# ==========================================================

with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    FEATURE_NAMES = json.load(
        file
    )


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)


# ==========================================================
# PREDICT CROP
# ==========================================================

def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    # ======================================================
    # CREATE INPUT
    # ======================================================

    input_data = np.array([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]])


    # ======================================================
    # PREDICTION
    # ======================================================

    prediction = model.predict(
        input_data
    )


    # ======================================================
    # PREDICTED CLASS
    # ======================================================

    predicted_index = int(
        prediction[0]
    )


    # ======================================================
    # CROP NAME
    # ======================================================

    predicted_crop = (
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    )


    # ======================================================
    # CONFIDENCE
    # ======================================================

    confidence = 0.0

    probabilities = None


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )


        confidence = (
            float(
                np.max(
                    probabilities
                )
            ) * 100
        )


    # ======================================================
    # TOP 3 PREDICTIONS
    # ======================================================

    top_predictions = []


    if probabilities is not None:

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]


        for index in top_indices:

            crop_name = (
                label_encoder.inverse_transform(
                    [int(index)]
                )[0]
            )


            top_predictions.append({

                "crop":
                    str(crop_name),

                "confidence":
                    round(
                        float(
                            probabilities[index]
                        ) * 100,
                        2
                    )

            })


    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {

        "crop":
            str(predicted_crop),

        "confidence":
            round(
                confidence,
                2
            ),

        "class_index":
            predicted_index,

        "top_predictions":
            top_predictions

    }