from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import DEFAULT_MODEL_PATH
from src.explainability.human_explanations import build_human_explanation
from src.preprocessing.data_loader import normalize_maintenance_columns, validate_required_features
from src.preprocessing.preprocessor import add_engineered_features
from src.utils.io import load_model, model_path_candidates


app = FastAPI(
    title="Explainable Predictive Maintenance API",
    description="Failure prediction and explanation service for industrial equipment.",
    version="1.0.0",
)


class SensorReading(BaseModel):
    machine_type: str = Field(..., examples=["M"])
    air_temperature: float = Field(..., examples=[298.1])
    process_temperature: float = Field(..., examples=[308.6])
    rotational_speed: float = Field(..., examples=[1551])
    torque: float = Field(..., examples=[42.8])
    tool_wear: float = Field(..., examples=[120])


def _get_model():
    for path in model_path_candidates(DEFAULT_MODEL_PATH):
        if path.exists():
            return load_model(path)
    raise HTTPException(
        status_code=503,
        detail="Model artifact not found. Train the model before serving predictions.",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: SensorReading) -> dict[str, object]:
    model = _get_model()
    row = pd.DataFrame([reading.model_dump()])
    row = add_engineered_features(row)
    prediction = model.predict(row)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0].max())

    status = "failure_risk" if int(prediction) == 1 else "normal"
    return {
        "prediction": int(prediction),
        "status": status,
        "confidence": probability,
        "explanation": build_human_explanation(row.iloc[0]),
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict[str, object]:
    model = _get_model()
    suffix = Path(file.filename or "upload.csv").suffix
    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        data = normalize_maintenance_columns(pd.read_csv(tmp.name))

    validate_required_features(data)
    features = add_engineered_features(data)
    predictions = model.predict(features)
    probabilities = (
        model.predict_proba(features).max(axis=1).tolist()
        if hasattr(model, "predict_proba")
        else [None] * len(features)
    )
    return {
        "rows": len(data),
        "predictions": [
            {
                "row": idx,
                "prediction": int(pred),
                "status": "failure_risk" if int(pred) == 1 else "normal",
                "confidence": prob,
            }
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities))
        ],
    }
