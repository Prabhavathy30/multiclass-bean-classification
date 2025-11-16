import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Bean Type Classifier", page_icon="🌱")

st.title("🌱 Bean Type Classifier")
st.write("Predict the type of bean based on input features.")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("bean_model_ml.joblib")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Inputs
feature_names = [
    "Area","Perimeter","MajorAxisLength","MinorAxisLength",
    "ConvexArea","EquivDiameter","Eccentricity","Solidity",
    "Extent","AspectRatio","Roundness"
]

inputs = [st.number_input(f"{f}", value=0.0) for f in feature_names]
input_array = np.array(inputs).reshape(1, -1)

# Predict button
if st.button("Predict"):
    if model:
        prediction = model.predict(input_array)[0]
        st.success(f"✅ Predicted Bean Type: **{prediction}**")
    else:
        st.warning("Model is not loaded. Check your joblib file and requirements.")
