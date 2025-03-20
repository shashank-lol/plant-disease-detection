from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json

app = FastAPI()

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Class Indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Image Preprocessing
def load_and_preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# API Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_array = load_and_preprocess_image(image_bytes)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()  # Run inference

        # Get prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_indices[str(predicted_class_index)]

        return {"prediction": predicted_class_name, "confidence": float(predictions[0][predicted_class_index])}

    except Exception as e:
        return {"error": str(e)}
