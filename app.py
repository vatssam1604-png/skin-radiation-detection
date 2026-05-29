from flask import Flask, request, render_template, jsonify, abort
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API Key for security
API_KEY = os.environ.get('API_KEY', 'your-secret-api-key-change-this')

# Load model components
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

def require_api_key(func):
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if api_key != API_KEY:
            abort(401, description="Invalid or missing API key")
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    age = float(data['age'])
    sex = data['sex']
    localization = data['localization']

    # Create input DataFrame
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'localization': [localization]
    })

    # Preprocess
    input_processed = pd.get_dummies(input_df, columns=['sex', 'localization'])

    # Ensure all columns are present
    for col in features:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Reorder columns
    input_processed = input_processed[features]

    # Scale age
    input_processed[['age']] = scaler.transform(input_processed[['age']])

    # Predict
    prediction = model.predict(input_processed)[0]
    probabilities = model.predict_proba(input_processed)[0]

    diagnosis = label_encoder.inverse_transform([prediction])[0]

    # Prepare probabilities dict
    prob_dict = {cls: float(prob) for cls, prob in zip(label_encoder.classes_, probabilities)}

    return render_template('results.html', diagnosis=diagnosis, probabilities=prob_dict, input_data=data)

@app.route('/api/predict', methods=['POST'])
@require_api_key
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['age', 'sex', 'localization']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        age = float(data['age'])
        sex = data['sex']
        localization = data['localization']

        # Validate inputs
        if age < 0 or age > 120:
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if sex not in ['male', 'female']:
            return jsonify({'error': 'Sex must be male or female'}), 400
        valid_localizations = ['face', 'trunk', 'lower extremity', 'upper extremity', 'abdomen', 'back', 'chest', 'foot', 'hand', 'neck', 'scalp', 'ear', 'genital', 'acral']
        if localization not in valid_localizations:
            return jsonify({'error': f'Invalid localization. Must be one of: {", ".join(valid_localizations)}'}), 400

        # Create input DataFrame
        input_df = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'localization': [localization]
        })

        # Preprocess
        input_processed = pd.get_dummies(input_df, columns=['sex', 'localization'])

        # Ensure all columns are present
        for col in features:
            if col not in input_processed.columns:
                input_processed[col] = 0

        # Reorder columns
        input_processed = input_processed[features]

        # Scale age
        input_processed[['age']] = scaler.transform(input_processed[['age']])

        # Predict
        prediction = model.predict(input_processed)[0]
        probabilities = model.predict_proba(input_processed)[0]

        diagnosis = label_encoder.inverse_transform([prediction])[0]

        # Prepare response
        response = {
            'diagnosis': diagnosis,
            'confidence': float(probabilities[prediction]),
            'probabilities': {cls: float(prob) for cls, prob in zip(label_encoder.classes_, probabilities)},
            'input': {
                'age': age,
                'sex': sex,
                'localization': localization
            }
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'classes': list(label_encoder.classes_),
        'features': features
    })

@app.route('/api/info', methods=['GET'])
@require_api_key
def model_info():
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'accuracy': 0.706,  # Approximate from training
        'classes': list(label_encoder.classes_),
        'features': features,
        'supported_localizations': ['face', 'trunk', 'lower extremity', 'upper extremity', 'abdomen', 'back', 'chest', 'foot', 'hand', 'neck', 'scalp', 'ear', 'genital', 'acral']
    })

if __name__ == '__main__':
    print("Starting Skin Lesion Classifier API Server...")
    print(f"API Key: {API_KEY}")
    print("Web interface: http://localhost:8000")
    print("API endpoints:")
    print("  POST /api/predict (requires X-API-Key header)")
    print("  GET /api/health")
    print("  GET /api/info (requires X-API-Key header)")
    app.run(host='0.0.0.0', port=8000, debug=True)