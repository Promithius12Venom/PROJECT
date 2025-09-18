import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import cv2

# Load the trained model once
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'
model = load_model(MODEL_SAVE_PATH)

class_names = ['Good', 'Bad-Blur', 'Bad-LowContrast']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image_array = img_to_array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def simple_blur_detection(image: Image.Image) -> bool:
    img_gray = np.array(image.convert('L'))
    variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return variance < 100

def simple_contrast_detection(image: Image.Image) -> bool:
    img_gray = np.array(image.convert('L'))
    stddev = img_gray.std()
    return stddev < 40

st.title("Image Quality Analyzer - Multi Image Upload (Improved)")

uploaded_files = st.file_uploader(
    "Upload one or more images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        input_array = preprocess_image(image)
        prediction_prob = model.predict(input_array)
        predicted_class = np.argmax(prediction_prob, axis=1)[0]
        confidence = prediction_prob[0][predicted_class]

        is_blur = simple_blur_detection(image)
        is_low_contrast = simple_contrast_detection(image)

        if confidence < 0.7:
            if is_blur:
                predicted_class = 1  # Bad-Blur
                confidence = 0.9
            elif is_low_contrast:
                predicted_class = 2  # Bad-LowContrast
                confidence = 0.9

        st.markdown(f"**Prediction:** {class_names[predicted_class]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        st.markdown("---")

else:
    st.info("Please upload at least one image to analyze.")
