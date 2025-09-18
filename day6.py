import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model - ensure this is the correct file path
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'
model = load_model(MODEL_SAVE_PATH)

# Class labels
class_names = ['Good', 'Bad-Blur', 'Bad-LowContrast']

st.title("Image Quality Analyzer")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load image and show preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess: resize and normalize
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)    # Batch dimension

    # Debug: show input shape and data range
    st.write(f"Input shape: {img_array.shape}")
    st.write(f"Pixel value range: min={img_array.min()}, max={img_array.max()}")

    # Predict probabilities
    prediction_prob = model.predict(img_array)
    
    # Debug: raw prediction output
    st.write("Model raw output (probabilities):", prediction_prob)

    # Highest probability class and confidence
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    confidence = prediction_prob[0][predicted_class]

    # Display class label and confidence nicely
    st.markdown(f"### Prediction: {class_names[predicted_class]}")
    st.markdown(f"### Confidence: {confidence:.2%}")
