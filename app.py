import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# ----------------------------
# 1️⃣ Load model with caching
# ----------------------------
@st.cache_resource
def load_cats_dogs_model():
    model = load_model("model/mobilenet_cats_dogs.h5")
    return model

model = load_cats_dogs_model()

# ----------------------------
# 2️⃣ App title
# ----------------------------
st.title("🐶🐱 Cats vs Dogs Classifier")
st.write("Upload your own image or pick a sample image to see predictions.")

# ----------------------------
# 3️⃣ Option to choose prediction mode
# ----------------------------
mode = st.radio("Select input mode:", ["Upload an image", "Use sample image"])

# ----------------------------
# 4️⃣ Upload mode
# ----------------------------
if mode == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_names = ["Cat", "Dog"]
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
        confidence = prediction[0][0] if predicted_class == "Dog" else 1 - prediction[0][0]

        st.success(f"Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")

# ----------------------------
# 5️⃣ Sample image mode
# ----------------------------
elif mode == "Use sample image":
    sample_dir = "sample_images"
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not sample_files:
        st.warning("No sample images found in the `sample_images/` folder.")
    else:
        selected_sample = st.selectbox("Choose a sample image:", sample_files)
        sample_path = os.path.join(sample_dir, selected_sample)
        
        image = Image.open(sample_path).resize((224, 224))
        st.image(image, caption=f"Sample Image: {selected_sample}", use_column_width=True)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_names = ["Cat", "Dog"]
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
        confidence = prediction[0][0] if predicted_class == "Dog" else 1 - prediction[0][0]

        st.success(f"Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
