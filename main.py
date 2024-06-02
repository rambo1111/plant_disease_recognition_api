from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins, you can specify specific origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the saved model
model = load_model("plant_disease_model.h5")
img_size = (128, 128)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read the image file as bytes
    contents = await file.read()
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(contents))
    # Resize the image
    image = image.resize(img_size)
    # Convert image to numpy array
    img = np.array(image) / 255.0  # Normalize pixel values
    
    # Make prediction using the model
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = ['Healthy', 'Powdery', 'Rust']
    predicted_label = class_labels[predicted_class[0]]
    
    # Format and return the prediction result
    return {predicted_label}
