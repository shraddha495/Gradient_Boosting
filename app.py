import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Gradient Boosting Prediction Portal", page_icon="📊", layout="centered")

st.title("📊 Gradient Boosting Prediction Portal")

# Robust model loader checking multiple common paths
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check possible locations for the model file
    possible_paths = [
        os.path.join(base_dir, 'gradientboosting.pkl'),
        os.path.join(base_dir, 'models', 'gradientboosting.pkl'),
        os.path.join(base_dir, 'model', 'gradientboosting.pkl')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("""
    ⚠️ **Model file not found!** 
    `gradientboosting.pkl` could not be located. 
    * **Fix:** Please ensure the `.pkl` file is uploaded to your GitHub repository in the same folder as `app.py` (or inside a `models/` folder).
    """)
else:
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        review = st.selectbox("Review", ["Positive", "Neutral", "Negative"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        try:
            features = np.array([[age, gender, review, education]], dtype=object)
            prediction = model.predict(features)[0]
            st.markdown("---")
            st.success(f"### Predicted Outcome: **{prediction}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
