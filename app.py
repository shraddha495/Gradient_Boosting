import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Gradient Boosting Prediction Portal", page_icon="📊", layout="centered")

st.title("📊 Gradient Boosting Prediction Portal")

# Robust model loader with automatic fallback
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'gradientboosting.pkl')
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), True  # True means real model loaded successfully
    else:
        return None, False  # False triggers fallback mode

model, is_real_model = load_model()

# Show status banner
if not is_real_model:
    st.warning("""
    ⚠️ **`gradientboosting.pkl` was not found in your GitHub repository!**
    * The app is currently running in **Fallback Mode** so it doesn't crash.
    * **To fix this permanently:** Upload your `gradientboosting.pkl` file directly to your GitHub repository in the exact same folder as `app.py`.
    """)
else:
    st.success("✅ Real model `gradientboosting.pkl` loaded successfully!")

# Input Form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    review = st.selectbox("Review", ["Positive", "Neutral", "Negative"])
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    if is_real_model:
        try:
            # Package features for your actual model
            features = np.array([[age, gender, review, education]], dtype=object)
            prediction = model.predict(features)[0]
            st.markdown("---")
            st.success(f"### Predicted Outcome: **{prediction}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        # Mock prediction output when the real model is missing
        st.markdown("---")
        st.info("ℹ️ *This is a mock prediction because your `.pkl` file is missing from GitHub.*")
        st.success("### Predicted Outcome: **Sample Result (Fallback Mode)**")
