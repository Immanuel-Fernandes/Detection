from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import io

# Load the trained model
model = load_model('plant_disease_detection_model.h5')
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Natural remedies for different diseases
recommendations = {
    'Healthy': "Your plant is healthy! Keep up the good care and ensure it receives proper sunlight, water, and nutrients.",
    'Powdery': (
        "It seems your plant has Powdery Mildew. You can treat it using natural remedies like:\n"
        "- Neem oil spray: Mix 2 tablespoons of neem oil with 1 gallon of water, and spray it on the affected areas.\n"
        "- Baking soda solution: Mix 1 tablespoon of baking soda with 1 gallon of water and spray the plant.\n"
        "- Garlic spray: Blend garlic with water and strain it to create a garlic-based natural pesticide.\n"
        "- Ensure good air circulation and avoid over-watering, as powdery mildew thrives in damp, humid conditions."
    ),
    'Rust': (
        "Your plant has Rust disease. You can treat it using natural remedies such as:\n"
        "- Garlic and chili spray: Blend garlic and chili with water to make a potent, natural insect repellent.\n"
        "- Baking soda solution: Mix 1 tablespoon of baking soda with 1 gallon of water and spray on infected areas.\n"
        "- Apple cider vinegar: Dilute apple cider vinegar with water and spray on affected areas.\n"
        "- Remove infected leaves and ensure proper plant spacing to improve air circulation, which will help prevent further spread."
    )
}

# FastAPI app instance
app = FastAPI()

# Function to preprocess the uploaded image and make predictions
def get_result(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    
    # Ensure the input shape matches the model's expectations
    assert x.shape == (1, 225, 225, 3), f"Expected input shape (1, 225, 225, 3), but got {x.shape}"
    
    predictions = model.predict(x)[0]
    return predictions

@app.post("/Plant_Disease_Detection_Predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()

        # Get predictions
        predictions = get_result(image_bytes)
        predicted_label = labels[np.argmax(predictions)]

        # Get recommendation based on the prediction
        recommendation = recommendations[predicted_label]

        # Return the prediction and recommendation as a JSON response
        return JSONResponse(content={
            "prediction": predicted_label,
            "recommendation": recommendation
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Detection API! Please upload an image using the '/Plant_Disease_Detection_Predict/' endpoint."}
