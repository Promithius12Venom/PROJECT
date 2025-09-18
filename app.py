import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
 

# Load model once
MODEL_SAVE_PATH = 'image_quality_analyzer_model.h5'
model = load_model(MODEL_SAVE_PATH)

class_names = ['Good', 'Bad-Blur', 'Bad-LowContrast']

# Set page config
st.set_page_config(
    page_title="Image Quality Analyzer",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Main app title and description
st.title("Image Quality Analyzer 🖼️")
st.write("Upload images below to analyze their quality based on a deep learning CNN classifier.")

# File uploader supports multiple files
uploaded_files = st.file_uploader(
    "Choose image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for image_file in uploaded_files:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {image_file.name}", use_column_width=True)

        # Preprocess image
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        pred_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][pred_class_idx]
        pred_label = class_names[pred_class_idx]

        # Show results with styling
        st.markdown(f"**Prediction:** {pred_label}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        st.markdown("---")
else:
    st.info("Please upload one or more images to start analysis.")

# Footer
st.markdown("""
---
© 2025 ANURAG KUMAR | AI/ML Engineer
""")
