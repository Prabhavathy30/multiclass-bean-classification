import streamlit as st
import numpy as np
import joblib  # Loaded from requirements.txt automatically

st.set_page_config(page_title="Bean Type Classifier", page_icon="🌱", layout="centered")

st.title("🌱 Bean Type Classifier")
st.write("Predict the type of bean based on input features.")

# ------------------------------
# Load the saved model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("bean_model_ml.joblib")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ------------------------------
# Feature Inputs
# ------------------------------
feature_names = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
    "Eccentricity",
    "Solidity",
    "Extent",
    "AspectRatio",
    "Roundness"
]

st.header("Enter Bean Features")
inputs = [st.number_input(f"{f}", value=0.0) for f in feature_names]
input_array = np.array(inputs).reshape(1, -1)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    if model:
        prediction = model.predict(input_array)[0]
        st.success(f"✅ Predicted Bean Type: **{prediction}**")
    else:
        st.warning("Model is not loaded. Please check the joblib file and requirements.txt.")
