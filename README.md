# Skin Lesion Classifier Web App & API

A Flask web application and REST API for predicting skin lesion diagnoses with a trained machine learning model. The production app is hosted on Vercel.

**Live web app:** [https://skin-radiation-detection.vercel.app/](https://skin-radiation-detection.vercel.app/)

## Project Overview

This repository includes:
- A browser-based prediction UI at [https://skin-radiation-detection.vercel.app/](https://skin-radiation-detection.vercel.app/)
- A secured JSON API for predictions
- Health and metadata endpoints
- Serialized model artifacts for inference

## Prerequisites

- Python 3.8 or newer
- Virtual environment recommended
- Project files should be in the repository root
- Dependencies listed in `requirements.txt` (see [Requirements](#requirements))

## Setup & Run

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies from `requirements.txt`:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Set your API key:
   ```bash
   export API_KEY=your-secret-api-key
   ```
4. Start the server:
   ```bash
   python app.py
   ```
5. Open the local web app:
   ```text
   http://localhost:8000
   ```

The hosted app is always available at [https://skin-radiation-detection.vercel.app/](https://skin-radiation-detection.vercel.app/).

## Web Interface

Use the web form at [https://skin-radiation-detection.vercel.app/](https://skin-radiation-detection.vercel.app/) to submit `age`, `sex`, and `localization` values and receive a diagnosis prediction. The same form is available locally at `http://localhost:8000` when you run `python app.py`.

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
- `requirements.txt` — Python packages for local setup and Vercel
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

Production:
```text
https://skin-radiation-detection.vercel.app/
```

## Notes

- If model artifacts are missing, run `python ml.py`.
- Keep `API_KEY` secure in environment variables (locally and in the Vercel project).


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

## Requirements

Python dependencies are defined in `requirements.txt`. Vercel reads this file on every deploy and installs the same packages used locally.

| Package | Role |
| --- | --- |
| `Flask` | Web app, templates, and REST routes |
| `flask-cors` | Cross-origin API access from the browser |
| `pandas`, `numpy` | Feature encoding and numeric input |
| `scikit-learn` | Loading and running the trained classifier |
| `matplotlib`, `seaborn` | Optional plotting used in the training pipeline |
| `gunicorn` | WSGI server for non-Vercel production hosts |
| `python-dotenv` | Loading local `.env` values such as `API_KEY` |

Install everything with:

```bash
python -m pip install -r requirements.txt
```

Pillow is commented out in `requirements.txt` and is only needed if you train or process images on the server.

## Deployment on Vercel

The app is deployed at [https://skin-radiation-detection.vercel.app/](https://skin-radiation-detection.vercel.app/). Vercel builds from this Git repository, installs packages from `requirements.txt`, and serves the Flask app over HTTPS.

To deploy or update:

1. Push the project to GitHub (or another Git remote Vercel can access).
2. In [Vercel](https://vercel.com/), import the repository and create a new project.
3. Leave the root directory as the repo root so Vercel can find `app.py`, `requirements.txt`, templates, static files, and model artifacts (`model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `features.pkl`).
4. Add environment variables in the Vercel project settings, including a strong `API_KEY`. Do not commit secrets.
5. Deploy. Vercel installs `requirements.txt`, then routes traffic to the Flask app.
6. After a successful build, open [https://skin-radiation-detection.vercel.app/](https://skin-radiation-detection.vercel.app/) to use the prediction form.

Subsequent pushes to the connected branch trigger a new Vercel deployment. HTTPS, the public hostname, and CDN caching are handled by Vercel.

API calls against production use the same paths as local development, for example:

```bash
curl -X POST https://skin-radiation-detection.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"age": 42, "sex": "female", "localization": "face"}'
```

## Files

- `app.py`: Flask server with web interface and API
- `ml.py`: Machine learning training script
- `requirements.txt`: Python dependencies for local install and Vercel
- `templates/index.html`: Web interface
- `templates/results.html`: Results page
- `static/css/style.css`: Styling
- `static/js/script.js`: Client-side JavaScript
- Model files: `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `features.pkl`