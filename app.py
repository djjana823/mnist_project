import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Digit Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28,28))
    
    img = np.array(image)
    img = img.reshape(1, -1)
    
    # Scale the input
    if hasattr(scaler, "feature_names_in_"):
        import pandas as pd
        img_df = pd.DataFrame(img, columns=scaler.feature_names_in_)
        img = scaler.transform(img_df)
    else:
        img = scaler.transform(img)
    
    prediction = model.predict(img)
    
    st.image(image, width=150)
    st.write("Predicted Digit:", prediction[0])