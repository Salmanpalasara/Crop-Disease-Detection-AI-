import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from .config import (
    FINAL_MODEL_PATH,
    IMAGE_SIZE,
)

# ==========================================================
# Load Model
# ==========================================================

model = tf.keras.models.load_model(
    FINAL_MODEL_PATH
)


# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = [

    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# # ==========================================================
# # Prediction Function
# # ==========================================================

# def predict_crop_disease(image_path):

#     img = image.load_img(

#         image_path,

#         target_size=IMAGE_SIZE

#     )

#     img_array = image.img_to_array(img)

#     img_array = np.expand_dims(
#         img_array,
#         axis=0
#     )

#     predictions = model.predict(
#         img_array,
#         verbose=0
#     )

#     predicted_index = int(
#         np.argmax(predictions[0])
#     )

#     confidence = float(
#         np.max(predictions[0])
#     )

#     predicted_class = CLASS_NAMES[
#         predicted_index
#     ]

#     return {

#         "class_index":
#             predicted_index,

#         "class_name":
#             predicted_class,

#         "disease_name":
#             predicted_class,

#         "confidence":
#             round(
#                 confidence * 100,
#                 2
#             ),

#         "top_predictions":
#             []

#     }

def predict_crop_disease(image_path):

    # ======================================================
    # Load Image
    # ======================================================

    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # ======================================================
    # Convert Image
    # ======================================================

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ======================================================
    # Prediction
    # ======================================================

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # ======================================================
    # Best Prediction
    # ======================================================

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # ======================================================
    # Top 3 Predictions
    # ======================================================

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({

            "class_name":
                CLASS_NAMES[index],

            "confidence":
                round(
                    float(
                        predictions[index] * 100
                    ),
                    2
                )

        })

    # ======================================================
    # Debug
    # ======================================================

    print("=" * 60)

    print("CROP DISEASE PREDICTION")

    print(
        "Predicted Class:",
        predicted_class
    )

    print(
        "Confidence:",
        round(
            confidence * 100,
            2
        ),
        "%"
    )

    print(
        "Top Predictions:",
        top_predictions
    )

    print("=" * 60)

    # ======================================================
    # Return
    # ======================================================

    return {

        "class_index":
            predicted_index,

        "class_name":
            predicted_class,

        "disease_name":
            predicted_class,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "top_predictions":
            top_predictions

    }