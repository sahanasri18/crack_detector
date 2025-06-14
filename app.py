import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model
model = joblib.load("model/rf_model.joblib")

st.title("🧱 Crack Detection on Surface Images")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file).resize((64, 64)).convert("L")
    img_array = np.array(image).flatten().reshape(1, -1) / 255.0

    # Predict
    proba = model.predict_proba(img_array)[0]
    pred = np.argmax(proba)
    confidence = round(np.max(proba) * 100, 2)

    # Output
    if pred == 1:
        st.error(f"⚠️ Crack Detected — Confidence: {confidence}%")
    else:
        st.success(f"✅ No Crack Detected — Confidence: {confidence}%")
