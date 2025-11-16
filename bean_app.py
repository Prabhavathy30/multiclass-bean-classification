import joblib

@st.cache_resource
def load_model():
    try:
        model = joblib.load("bean_model_ml.joblib")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
