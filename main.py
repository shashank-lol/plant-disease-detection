from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
import gdown

url = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
output = "model.h5"
gdown.download(url, output, quiet=False)

app = FastAPI()

# Load Model and Class Indices
model = tf.keras.models.load_model(output)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Image Preprocessing
def load_and_preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# API Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()  # Read the image as bytes
        img_array = load_and_preprocess_image(image_bytes)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]
        return {"prediction": predicted_class_name}
    except Exception as e:
        return {"error": str(e)}


# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import json

# # Load Model
# model = tf.keras.models.load_model("model.h5")

# # Load Class Indices
# with open("class_indices.json", "r") as f:
#     class_indices = json.load(f)

# def preprocess_image(image_path: str) -> np.ndarray:
#     """Load and preprocess an image from local storage."""
#     img = Image.open(image_path).convert("RGB")  # Convert to RGB
#     img = img.resize((224, 224))  # Resize to match model input size
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# def predict(image_path: str):
#     """Make a prediction for a given local image."""
#     preprocessed_img = preprocess_image(image_path)
#     predictions = model.predict(preprocessed_img)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class_name = class_indices[str(predicted_class_index)]
#     return predicted_class_name

# # 🔥 Test with a local image
# image_path = "test.jpg"  # Change this to the path of your image
# prediction = predict(image_path)
# print(f"Prediction: {prediction}")
