# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import uvicorn

app = FastAPI()

# Load your model
model = load_model("skin_cancer_cnn_model.h5")

# Define a function to preprocess the image
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    prediction = model.predict(img)
    result = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer"
    return JSONResponse({"prediction": result})
