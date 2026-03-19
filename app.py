import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Digit Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28,28))
    
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, -1)
    
    prediction = model.predict(img)
    
    st.image(image, width=150)
    st.write("Predicted Digit:", prediction[0])