import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model function
@st.cache_resource
def load_pneumonia_model():
    model_path = "models/imageclassifier.h5"  # Update if necessary
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    if image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make prediction
def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_label = "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia"
    return class_label

# Main app
def main():
    st.set_page_config(page_title="Pneumonia Detection", layout="centered")

    # Custom Styling
    st.markdown(
        """
        <style>
            .main { background-color: #f4f4f4; }
            h1 { color: #1E3A8A; text-align: center; font-size: 28px; }
            .stButton>button { background-color: #1E40AF; color: white; font-size: 16px; }
            .image-container { display: flex; align-items: center; justify-content: center; gap: 20px; }
            .image-container img { width: 300px; border-radius: 8px; }
            .result-box { padding: 15px; background-color: white; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); text-align: center; margin-top: 10px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App Title
    st.markdown("<h1>Pneumonia Detection from X-ray Images</h1>", unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])

    # Layout for image & button side by side
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Creating two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded X-ray Image", width=300)  # Reduced size

        with col2:
            st.markdown("### Run Diagnosis")
            run_button = st.button("🔍 Analyze X-ray", use_container_width=True)

            # Load Model
            model = load_pneumonia_model()

            # Run diagnosis when button is clicked and display result immediately below
            if run_button:
                with st.spinner("Running prediction..."):
                    result = predict(image, model)

                # Display results directly below the button
                st.markdown(
                    f'<div class="result-box"><h2>{result}</h2></div>',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
