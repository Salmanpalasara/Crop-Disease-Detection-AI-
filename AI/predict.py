import json

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from .config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
)


# ==========================================================
# Load Model
# ==========================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================================
# Load Class Names
# ==========================================================

with open(

    CLASS_NAMES_PATH,

    "r",

    encoding="utf-8"

) as file:

    CLASS_NAMES = json.load(
        file
    )


# ==========================================================
# Prediction Function
# ==========================================================

def predict_soil(image_path):

    # ======================================================
    # LOAD IMAGE
    # ======================================================

    img = image.load_img(

        image_path,

        target_size=IMAGE_SIZE

    )


    # ======================================================
    # IMAGE TO ARRAY
    # ======================================================

    img_array = image.img_to_array(
        img
    )


    # ======================================================
    # ADD BATCH DIMENSION
    # ======================================================

    img_array = np.expand_dims(

        img_array,

        axis=0

    )


    # ======================================================
    # PREDICTION
    # ======================================================

    predictions = model.predict(

        img_array,

        verbose=0

    )


    probabilities = predictions[0]


    # ======================================================
    # BEST CLASS
    # ======================================================

    predicted_index = int(

        np.argmax(
            probabilities
        )

    )


    confidence = float(

        probabilities[
            predicted_index
        ]

    )


    # ======================================================
    # CLASS NAME
    # ======================================================

    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    # ======================================================
    # TOP 3
    # ======================================================

    top_indices = np.argsort(

        probabilities

    )[::-1][:3]


    top_predictions = []


    for index in top_indices:

        index = int(index)

        top_predictions.append({

            "class_name":
                CLASS_NAMES[index],

            "confidence":
                round(

                    float(
                        probabilities[index]
                    ) * 100,

                    2

                )

        })


    # ======================================================
    # RESULT
    # ======================================================

    return {

        "class_index":
            predicted_index,

        "class_name":
            predicted_class,

        "soil_type":
            predicted_class,

        "confidence":
            round(

                confidence * 100,

                2

            ),

        "top_predictions":
            top_predictions

    }