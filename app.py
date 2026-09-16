import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load your trained Gradient Boosting model
with open('gradientboosting.pkl', 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.4);
            max-width: 480px;
            width: 100%;
            backdrop-filter: blur(10px);
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h2 {
            color: #2d3748;
            text-align: center;
            margin-bottom: 25px;
            font-weight: 700;
            font-size: 26px;
            letter-spacing: -0.5px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #4a5568;
            font-weight: 500;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #cbd5e0;
            border-radius: 10px;
            font-size: 15px;
            background-color: #f7fafc;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        }
        input:focus, select:focus {
            border-color: #667eea;
            background-color: #fff;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), inset 0 2px 4px rgba(0,0,0,0.02);
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            margin-top: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        button:active {
            transform: translateY(0);
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            background: #edf2f7;
            border-left: 5px solid #667eea;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            animation: slideUp 0.5s ease;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-card h3 {
            color: #2d3748;
            font-size: 16px;
            margin-bottom: 5px;
        }
        .result-value {
            color: #667eea;
            font-size: 22px;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Model Prediction</h2>
        <form method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required placeholder="Enter age">
            </div>
            
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="" disabled selected>Select Gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="review">Review</label>
                <select id="review" name="review" required>
                    <option value="" disabled selected>Select Review</option>
                    <option value="Positive">Positive</option>
                    <option value="Neutral">Neutral</option>
                    <option value="Negative">Negative</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="education">Education</label>
                <select id="education" name="education" required>
                    <option value="" disabled selected>Select Education Level</option>
                    <option value="High School">High School</option>
                    <option value="Bachelor">Bachelor</option>
                    <option value="Master">Master</option>
                    <option value="PhD">PhD</option>
                </select>
            </div>
            
            <button type="submit">Predict</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Predicted Outcome</h3>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Features extracted based on your model's requirement: ['age', 'gender', 'review', 'education']
            age = float(request.form['age'])
            gender = request.form['gender']
            review = request.form['review']
            education = request.form['education']
            
            # Note: If your model pipeline uses numerical encoding for categories, 
            # make sure to map the string inputs to their respective encoded numbers here.
            features = np.array([[age, gender, review, education]], dtype=object)
            
            pred = model.predict(features)[0]
            prediction = str(pred)
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
