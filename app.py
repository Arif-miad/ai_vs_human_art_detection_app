# ----------------------------
# IMPORT LIBRARIES
# ----------------------------
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image  

# ----------------------------
# LOAD MODEL
# ----------------------------
model_path = "ai_vs_human_art_cnn_model.h5"   # path to your saved model
model = load_model(model_path)

# ----------------------------
# APP TITLE
# ----------------------------
st.title("AI vs Human Art Detection")
st.write("Upload an image and the model will predict whether it is AI-generated or Human-created art.")

# ----------------------------
# IMAGE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    
    img = img.copy()
    img.thumbnail((img.width//2, img.height//2))
    
    # Display image
    st.image(img, caption='Uploaded Image', use_container_width=False)
    
    # Preprocess image for model
    img_resized = img.resize((128,128))
    img_array = image.img_to_array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_array)
    class_label = "AI Art" if prediction[0][0] < 0.5 else "Human Art"
    confidence = prediction[0][0] if prediction[0][0]<0.5 else 1-prediction[0][0]
    
    st.write(f"Prediction: **{class_label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
