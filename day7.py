import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Load model
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'
model = load_model(MODEL_SAVE_PATH)

class_names = ['Good', 'Bad-Blur', 'Bad-LowContrast']

st.title("Image Quality Analyzer - Enhanced UI")

st.write("""
Upload one or more images to analyze their quality.  
The app will show the uploaded image(s), predicted class, and confidence.
""")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)

        # Preprocess image
        image_resized = image.resize((224, 224))
        image_array = img_to_array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction_prob = model.predict(image_array)
        predicted_class = np.argmax(prediction_prob, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence = prediction_prob[0][predicted_class]

        # Display Prediction Results
        st.markdown(f"**Prediction:** {predicted_label}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        st.markdown("---")
else:
    st.info("Please upload at least one image.")
