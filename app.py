import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

@st.cache_resource
def load_model():
    # Uses relative path from app.py location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'CAT_DOG_MODEL.h5')
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

st.title("🐾 Cat vs Dog Classifier")
file = st.file_uploader("Upload a photo...", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (160, 160)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image.convert("RGB"))
    img_reshape = np.expand_dims(img_array, axis=0) / 255.0
    return model.predict(img_reshape)

if file:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    predictions = import_and_predict(image, model)

    score = float(predictions[0][0])

    if score > 0.5:
        st.success(f"It's a **Dog**! 🐶 (Confidence: {score*100:.2f}%)")
    else:
        st.success(f"It's a **Cat**! 🐱 (Confidence: {(1-score)*100:.2f}%)")