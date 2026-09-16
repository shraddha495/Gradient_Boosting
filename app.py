import streamlit as st
import pickle
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Prediction Portal", 
    page_icon="✨", 
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .block-container {
            background: rgba(255, 255, 255, 0.90);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 40px !important;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
            margin-top: 5vh;
            margin-bottom: 5vh;
        }
        h1 { color: #1a202c; text-align: center; font-weight: 800; }
        p.subtitle { text-align: center; color: #4a5568; font-size: 16px; margin-bottom: 25px; }
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border-radius: 12px;
            padding: 12px;
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .success-box {
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(72, 187, 120, 0.15);
            text-align: center;
            margin-top: 20px;
        }
        .debug-box {
            background: #edf2f7;
            border-left: 4px solid #4a5568;
            padding: 12px;
            font-family: monospace;
            font-size: 13px;
            border-radius: 4px;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Prediction Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Gradient Boosting Interactive Model</p>", unsafe_allow_html=True)

# Model Loader with Status Detection
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'gradientboosting.pkl')
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), True  # True = Real Model
    else:
        # Create a smarter fallback model where outputs dynamically change based on input math
        X_dummy = np.array([[10, 0, 0, 0], [50, 1, 1, 1], [30, 2, 2, 2], [70, 0, 1, 3]])
        y_dummy = np.array([0, 1, 0, 1])
        model = GradientBoostingClassifier()
        model.fit(X_dummy, y_dummy)
        return model, False  # False = Fallback Mode

model, is_real_model = load_model()

# Status notification banner
if is_real_model:
    st.success("✅ Connected to your **Real Model (`gradientboosting.pkl`)**")
else:
    st.warning("⚠️ Running on **Fallback Mode**. Upload `gradientboosting.pkl` to GitHub to use your actual model.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        review = st.selectbox("Review", ["Positive", "Neutral", "Negative"])
        
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Generate Prediction 🚀")

if submit_button:
    try:
        # Encoding categorical choices to numeric
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        review_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
        
        g_val = gender_map.get(gender, 0)
        r_val = review_map.get(review, 1)
        e_val = edu_map.get(education, 1)
        
        # Prepare feature array
        features = np.array([[age, g_val, r_val, e_val]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Display Result
        st.markdown(f"""
            <div class="success-box">
                <h4 style="color: #276749; margin-bottom: 5px;">Predicted Outcome</h4>
                <h2 style="color: #22543d; font-size: 28px;">{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Debug view to prove inputs are changing
        st.markdown(f"""
            <div class="debug-box">
                <b>🔍 Debug Info Sent to Model:</b><br>
                Features Array: {features.tolist()}
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
