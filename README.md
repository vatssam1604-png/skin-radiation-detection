# Skin Lesion Classifier Web App & API

A local Flask web application and REST API for predicting skin lesion diagnoses with a trained machine learning model.

## Project Overview

This repository includes:
- A browser-based prediction UI at `http://localhost:8000`
- A secured JSON API for predictions
- Health and metadata endpoints
- Serialized model artifacts for inference

## Prerequisites

- Python 3.8 or newer
- Virtual environment recommended
- Project files should be in the repository root

## Setup & Run

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install flask flask-cors pandas numpy scikit-learn
   ```
3. Set your API key:
   ```bash
   export API_KEY=your-secret-api-key
   ```
4. Start the server:
   ```bash
   python app.py
   ```
5. Open the web app:
   ```text
   http://localhost:8000
   ```

## Web Interface

Use the web form at `http://localhost:8000` to submit `age`, `sex`, and `localization` values and receive a diagnosis prediction.

## API Endpoints

### `POST /api/predict`
Predict skin lesion diagnosis.

- Authentication: required
- Header: `X-API-Key: <your-api-key>`
- Content-Type: `application/json`

Request body example:
```json
{
  "age": 42,
  "sex": "female",
  "localization": "face"
}
```

Response example:
```json
{
  "diagnosis": "nv",
  "confidence": 0.6884,
  "probabilities": {
    "akiec": 0.0000,
    "bcc": 0.0000,
    "bkl": 0.0000,
    "df": 0.0000,
    "mel": 0.0000,
    "nv": 0.6884,
    "vasc": 0.3116
  },
  "input": {
    "age": 42,
    "sex": "female",
    "localization": "face"
  }
}
```

Example cURL:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"age": 42, "sex": "female", "localization": "face"}'
```

### `GET /api/health`
Returns application status.

- Authentication: not required

Example response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
  "features": ["age", "sex_female", "sex_male", ...]
}
```

### `GET /api/info`
Returns model metadata and supported values.

- Authentication: required
- Header: `X-API-Key: <your-api-key>`

Example response:
```json
{
  "model_type": "RandomForestClassifier",
  "accuracy": 0.706,
  "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
  "features": ["age", "sex_female", "sex_male", ...],
  "supported_localizations": ["face", "trunk", "lower extremity", ...]
}
```

## Input Validation

The app validates inputs:
- `age`: number between `0` and `120`
- `sex`: `male` or `female`
- `localization`: supported skin location string

## Project Structure

- `app.py` — Flask web server and API
- `ml.py` — training/data pipeline script
- `templates/index.html` — main UI page
- `templates/results.html` — results display page
- `static/css/style.css` — UI styling
- `static/js/script.js` — optional browser script
- `model.pkl` — trained model
- `scaler.pkl` — preprocessing scaler
- `label_encoder.pkl` — label encoder
- `features.pkl` — feature ordering

## Running Locally

Browse to:
```text
http://localhost:8000
```

## Notes

- If model artifacts are missing, run `python ml.py`.
- Use `gunicorn` or another WSGI server for production.
- Keep `API_KEY` secure in environment variables.


## Usage Examples

### Python
```python
import requests

url = "http://your-server:8000/api/predict"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-secret-api-key"
}
data = {
    "age": 35,
    "sex": "male",
    "localization": "face"
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(f"Diagnosis: {result['diagnosis']}")
```

### JavaScript
```javascript
fetch('http://your-server:8000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-secret-api-key'
  },
  body: JSON.stringify({
    age: 35,
    sex: 'male',
    localization: 'face'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Deployment

For production deployment:
1. Set a strong API key
2. Use a production WSGI server (gunicorn, uwsgi)
3. Configure reverse proxy (nginx)
4. Enable HTTPS
5. Set proper environment variables

## Files

- `app.py`: Flask server with web interface and API
- `ml.py`: Machine learning training script
- `templates/index.html`: Web interface
- `templates/results.html`: Results page
- `static/css/style.css`: Styling
- `static/js/script.js`: Client-side JavaScript
- Model files: `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `features.pkl`