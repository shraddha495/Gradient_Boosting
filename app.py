import streamlit as st
import pickle
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Gradient Boosting Prediction Portal", 
    page_icon="📊", 
    layout="centered"
)

# Custom UI Styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a6fe0 0%, #684092 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Gradient Boosting Prediction Portal")
st.write("Fill in the details below to generate a prediction.")

# Robust model loading to prevent FileNotFoundError
@st.cache_resource
def load_model():
    # Safely locate the model file relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'gradientboosting.pkl')
    
    if not os.path.exists(model_path):
        return None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Graceful check if the model is missing
if model is None:
    st.error("⚠️ **Model file not found!** `gradientboosting.pkl` could not be located in your project directory. Please make sure it is uploaded to your GitHub repository.")
else:
    # Input Form
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        review = st.selectbox("Review", ["Positive", "Neutral", "Negative"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        try:
            # Package features into an array (ensure this matches your training pipeline format)
            features = np.array([[age, gender, review, education]], dtype=object)
            
            # Predict
            prediction = model.predict(features)[0]
            
            st.markdown("---")
            st.success(f"### Predicted Outcome: **{prediction}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
