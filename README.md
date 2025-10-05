# Exo: Exoplanet Detection Model

Exo is an AI-powered application and API for classifying Kepler Objects of Interest (KOI) as confirmed exoplanets, candidates, or false positives. It uses machine learning models to analyze features from Kepler's dataset and predict the likelihood of an object being an exoplanet.

## Features

- **Single Prediction:** Input KOI features manually to get a real-time prediction.
- **Batch Prediction:** Upload a CSV file for batch classification.
- **Model Selection:** Choose from available ML models (e.g., XGBoost, Ensemble).
- **REST API:** Predict via HTTP endpoints for integration and automation.
- **Interactive Web UI:** User-friendly interface for manual or batch prediction.

## Technologies Used

- **Python (FastAPI):** Backend API server.
- **HTML/CSS/JS:** Web frontend for user interaction.
- **Machine Learning:** Scikit-learn, XGBoost, joblib/pickle model management.
- **Docker:** Containerized deployment.
- **Pydantic:** Input validation and output modeling.

## Quick Start

### Running Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ikeasamoahansah/exo.git
   cd exo
   ```

2. **Build & Run with Docker**
   ```bash
   docker build -t exo .
   docker run -p 8000:8000 exo
   ```

3. **Manual Run (Python)**
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Open Web UI**
   - Load `index.html` in your browser and set API endpoint to `http://localhost:8000`.

### API Endpoints

- `GET /`: API info and available endpoints.
- `GET /models`: List available models.
- `GET /health`: Health check.
- `POST /predict`: Single KOI prediction.
- `POST /predict/batch`: Batch prediction via CSV upload.
- `POST /predict/batch/json`: Batch prediction via JSON.

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"koi_model_snr": 15.5, "koi_prad": 2.3, "koi_fpflag_ss": 0, "koi_fpflag_co": 0, "koi_period": 10.5, "koi_depth": 500.0, "koi_fpflag_nt": 1, "koi_insol": 1.2}'
```

## Input Features

| Feature           | Description                                |
|-------------------|--------------------------------------------|
| koi_model_snr     | Transit signal-to-noise ratio              |
| koi_prad          | Planetary radius (Earth radii)             |
| koi_fpflag_ss     | Stellar eclipse flag (0 or 1)              |
| koi_fpflag_co     | Centroid offset flag (0 or 1)              |
| koi_period        | Orbital period (days)                      |
| koi_depth         | Transit depth (ppm)                        |
| koi_fpflag_nt     | Not transit-like flag (0 or 1)             |
| koi_insol         | Insolation flux (Earth units)              |

## Model Output

- **prediction:** Integer code (1 = Confirmed Exoplanet, 2 = Candidate, 0 = False Positive)
- **probability:** Confidence score (0.0 - 1.0)
- **classification:** Human-readable label

## Batch Prediction

- Upload a CSV file with columns matching the input features above.
- Results are returned as a JSON array or downloadable CSV.

## License

MIT License

---

**Author:** ~ike

For questions or contributions, open an issue or pull request on [GitHub](https://github.com/ikeasamoahansah/exo).
````
