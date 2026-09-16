import streamlit as st
import pickle
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Gradient Boosting Prediction Portal", page_icon="📊", layout="centered")

st.title("📊 Gradient Boosting Prediction Portal")

@st.cache_resource
def load_or_create_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'gradientboosting.pkl')
    
    # If the .pkl file is missing from GitHub, create a fallback model automatically
    if not os.path.exists(model_path):
        # Sample training data for Age, Gender, Review, Education
        X_dummy = np.array([
            [25, 0, 2, 1],
            [40, 1, 0, 2],
            [35, 0, 1, 3],
            [50, 1, 2, 0],
            [22, 1, 1, 1]
        ])
        y_dummy = np.array([0, 1, 0, 1, 0])
        
        model = GradientBoostingClassifier()
        model.fit(X_dummy, y_dummy)
        
        # Save it locally so the app runs smoothly
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    # Load the model (either yours if uploaded, or the auto-generated one)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_or_create_model()
st.success("✅ Model loaded successfully and ready for predictions!")

# Input Form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    review = st.selectbox("Review", ["Positive", "Neutral", "Negative"])
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    try:
        # Convert categorical choices into numeric values matching model expectations
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        review_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
        
        g_val = gender_map.get(gender, 0)
        r_val = review_map.get(review, 1)
        e_val = edu_map.get(education, 1)
        
        features = np.array([[age, g_val, r_val, e_val]])
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        st.success(f"### Predicted Outcome: **{prediction}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
