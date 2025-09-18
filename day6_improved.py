import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import cv2

# Load the trained model
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'
model = load_model(MODEL_SAVE_PATH)

class_names = ['Good', 'Bad-Blur', 'Bad-LowContrast']

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to 224x224 as expected by model
    image = image.resize((224, 224))
    image_array = img_to_array(image).astype('float32') / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return image_array

def simple_blur_detection(image: Image.Image) -> bool:
    # Convert image to grayscale and compute variance of Laplacian (blur measure)
    img_gray = np.array(image.convert('L'))
    variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return variance < 100  # Threshold, adjust as needed

def simple_contrast_detection(image: Image.Image) -> bool:
    # Compute standard deviation of grayscale image pixels (contrast measure)
    img_gray = np.array(image.convert('L'))
    stddev = img_gray.std()
    return stddev < 40  # Threshold, adjust as needed

st.title("Image Quality Analyzer (Improved)")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model prediction
    input_array = preprocess_image(image)

    # Predict probabilities using the model
    prediction_prob = model.predict(input_array)
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    confidence = prediction_prob[0][predicted_class]

    # Apply heuristic checks if confidence is low
    is_blur = simple_blur_detection(image)
    is_low_contrast = simple_contrast_detection(image)

    if confidence < 0.7:
        if is_blur:
            predicted_class = 1  # Bad-Blur
            confidence = 0.9
        elif is_low_contrast:
            predicted_class = 2  # Bad-LowContrast
            confidence = 0.9

    st.markdown(f"### Prediction: {class_names[predicted_class]}")
    st.markdown(f"### Confidence: {confidence:.2%}")

    # Debug info (optional)
    st.write("Raw model output (probabilities):", prediction_prob)
    st.write(f"Blur detected (heuristic): {is_blur}")
    st.write(f"Low contrast detected (heuristic): {is_low_contrast}")
else:
    st.info("Please upload an image to analyze.")
