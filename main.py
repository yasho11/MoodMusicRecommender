import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = "emotion_detection_model.h5"
model = load_model(MODEL_PATH)

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))  # Resize to 48x48
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Function to predict emotion
def predict_emotion(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    emotion_index = np.argmax(predictions)
    confidence = predictions[0][emotion_index]
    return EMOTION_LABELS[emotion_index], confidence

# Streamlit app
st.title("Emotion Detection App")
st.write("Upload an image to detect the emotion!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict emotion
    with st.spinner("Predicting..."):
        emotion, confidence = predict_emotion(image)

    # Display the results
    st.success(f"Detected Emotion: **{emotion}**")
    st.write(f"Confidence: **{confidence:.2f}**")
