# Deployment Guide

## Local Research Run

1. Install dependencies.
2. Place the dataset in `data/raw/`.
3. Train models with `python main.py train --data data/raw/ai4i2020.csv --target failure_binary --tune`.
4. Review `reports/artifacts/model_comparison.csv`.
5. Review generated figures in `reports/figures/`.

## API Deployment

Run:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Use `POST /predict` for a single machine reading and `POST /predict/batch` for CSV scoring.

## Dashboard Deployment

Run:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard reads the trained model from `reports/models/best_model.joblib`.

## Docker Deployment

Build and run both services:

```bash
docker compose up --build
```

API: `http://localhost:8000`

Dashboard: `http://localhost:8501`

## Industry Deployment Pattern

For production, deploy the model service behind an internal API gateway. Connect the data layer to historian data, SCADA exports, IoT streams, or a batch data warehouse. Store predictions and explanations in a database for auditability. Connect high-risk predictions to an alert manager or computerized maintenance management system.

## Monitoring

Track:

- Sensor input drift
- Failure-rate drift
- Prediction confidence distribution
- False positive and false negative feedback
- Explanation stability
- API latency
- Model version and training dataset version

