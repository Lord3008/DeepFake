# Step 2: Import required libraries
import streamlit as st
from transformers import pipeline
from PIL import Image

# Step 3: Load the image classification pipeline
st.title("DeepFake Image Detector")
st.write(
    "Upload an image to detect if it's real or a deepfake using the DeepFake Detector model."
)

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model")

pipe = load_model()

# Step 4: File upload and prediction
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    st.write("Analyzing...")
    results = pipe(image)

    # Display results
    st.write("Results:")
    for result in results:
        st.write(f"**{result['label']}**: {result['score']:.4f}")
