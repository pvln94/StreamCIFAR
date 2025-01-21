# Import Packages
import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf

# Class names for CIFAR-10 dataset
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Create a function to load the saved model
@st.cache_resource
def load_my_model():
    model_path = r"C:\Users\naras\Downloads\StreamCIFAR-main\my_model.h5"  # Use raw string for Windows paths
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model
model = load_my_model()

# Streamlit App
st.title("Image Classification with CIFAR-10 Dataset")
st.header("Upload an image to classify it into one of the following categories:")
st.text(", ".join(class_names))  # Display class names

# File uploader for image input
file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])

# Function to process and predict the uploaded image
def import_and_predict(image_data, model):
    try:
        size = (32, 32)
        image = image_data.convert("RGB")  # Ensure the image is in RGB format
        image = ImageOps.fit(image, size, Image.LANCZOS)  # Use LANCZOS for better quality resizing
        img = np.asarray(image) / 255.0  # Normalize the image
        img_reshape = img[np.newaxis, ...]  # Add batch dimension
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Prediction Button
if st.button("Predict"):
    if file is not None:
        try:
            # Open and display the image
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict the class
            predictions = import_and_predict(image, model)
            if predictions is not None:
                pred_class = class_names[np.argmax(predictions)]
                st.success(f"The image is most likely a: {pred_class}")
            else:
                st.error("Prediction could not be made. Please try again.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload an image before predicting.")
