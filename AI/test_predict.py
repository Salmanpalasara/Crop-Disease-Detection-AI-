from pathlib import Path

from .predict import predict_soil


# ==========================================================
# TEST IMAGE
# ==========================================================

IMAGE_PATH = Path(
    r"C:\Users\Lenovo\Desktop\Crop Disease\Dataset\Soil_dataset\test\Red soil\Copy of 2560px-A_red_soil_crop_field.png"
)


# ==========================================================
# RUN PREDICTION
# ==========================================================

result = predict_soil(IMAGE_PATH)


# ==========================================================
# DISPLAY RESULT
# ==========================================================

print("=" * 60)
print("SOIL PREDICTION RESULT")
print("=" * 60)

print("Predicted Soil :", result["soil_type"])
print("Confidence     :", result["confidence"], "%")

print("\nTop 3 Predictions:")

for prediction in result["top_predictions"]:

    print(
        f"{prediction['class_name']} "
        f"-> {prediction['confidence']}%"
    )

print("=" * 60)