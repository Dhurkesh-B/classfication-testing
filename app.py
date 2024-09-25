import streamlit as st
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

# Load your Hugging Face model and feature extractor
model_id = "Dhurkesh1/tomatoDiseaseClassifier"  # Your Hugging Face model ID
model = AutoModelForImageClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# Define a function to process and classify the uploaded image
def predict_image_class(image):
    # Convert the image to RGB (if it's not already)
    image = image.convert("RGB")
    
    # Apply the feature extractor to the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class label
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Get the class name from the model config
    class_name = model.config.id2label[predicted_class_idx]
    
    return class_name

# Streamlit app
st.title("Plant Disease Classifier")
st.write("Upload an image of a plant leaf to predict its disease class.")

# File uploader to accept the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Run the image through the model and display the predicted class
    st.write("Classifying...")
    predicted_class = predict_image_class(image)
    st.write(f"Predicted Class: **{predicted_class}**")
