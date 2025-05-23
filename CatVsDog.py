import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model(r'C:\Users\DELL\Downloads\Cat_vs_Dog.h5')

# Function to process the img and make predictions
def predict_image(img):
    img = img.resize((160, 160))                # Resize to match model input
    img_array = np.array(img)                   # Convert to NumPy array
    img_array = img_array / 255.0               # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.markdown("<h1 style='text-align: center;'>🐱 Cat vs 🐶 Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image, and our deep learning model will predict whether it's a cat or a dog!</p>", unsafe_allow_html=True)

# File uploader widget
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "jfif"])

# If user uploads an image
if uploaded_image is not None:
    img = Image.open(uploaded_image)

    # Show the uploaded image
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing Image..."):
        prediction = predict_image(img)
        predicted_class = np.argmax(prediction)
  # Use sigmoid threshold

    # Display result
    if predicted_class == 1:
        st.success("✅ It's a **Dog**!")
    else:
        st.success("✅ It's a **Cat**!")

    # Show confidence
    confidence = float(prediction[0][0])
    st.info(f"Model Confidence: **{confidence:.2f}**")

else:
    st.info("Please upload a JPG/PNG image.")
