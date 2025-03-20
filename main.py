from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
import gdown
import os

app = FastAPI()

# Google Drive file ID for model.tflite
file_id = "14helETXqZ2JxyNscXWLay9QatB_uvBpu"  # Replace with your actual file ID
output = "model.tflite"

# Download model if not already present
if not os.path.exists(output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=output)
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
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# API Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()  # Read image
        img_array = load_and_preprocess_image(image_bytes)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]

        return {"prediction": predicted_class_name}
    except Exception as e:
        return {"error": str(e)}
