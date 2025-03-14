# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

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

# Function to preprocess the uploaded image and make predictions
def get_result(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    
    # Ensure the input shape matches the model's expectations
    assert x.shape == (1, 225, 225, 3), f"Expected input shape (1, 225, 225, 3), but got {x.shape}"
    
    predictions = model.predict(x)[0]
    return predictions

# Streamlit app layout
st.title("Plant Disease Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded file temporarily
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get predictions
        predictions = get_result("temp.jpg")
        predicted_label = labels[np.argmax(predictions)]

        # Display the prediction
        st.write(f"Prediction: {predicted_label}")

        # Display recommendation based on the prediction
        st.write("Recommendation: ")
        st.write(recommendations[predicted_label])

    except Exception as e:
        st.write("Error occurred: ", e)

# Display sample images for demonstration purposes
st.subheader("Sample Images for Demonstration")

col1, col2, col3 = st.columns(3)

def display_sample_image(image_path):
    try:
        # Get predictions for the sample image
        predictions = get_result(image_path)
        predicted_label = labels[np.argmax(predictions)]

        # Display the sample image and its prediction
        st.image(image_path, caption=f"{predicted_label} Image", use_column_width=True)
        st.write(f"Prediction: {predicted_label}")
        st.write("Recommendation: ")
        st.write(recommendations[predicted_label])
    except Exception as e:
        st.write("Error occurred: ", e)

if col1.button('Upload Healthy Sample', key='button1'):
    display_sample_image('sample_healthy.jpg')
with col1:
    st.image('sample_healthy.jpg', caption='Healthy', use_column_width=True)

if col2.button('Upload Powdery Sample', key='button2'):
    display_sample_image('sample_powdery.jpg')
with col2:
    st.image('sample_powdery.jpg', caption='Powdery', use_column_width=True)

if col3.button('Upload Rust Sample', key='button3'):
    display_sample_image('sample_rust.jpg')
with col3:
    st.image('sample_rust.jpg', caption='Rust', use_column_width=True)
