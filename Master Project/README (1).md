# Explainable Predictive Maintenance System using Machine Learning for Industrial Equipment

An industry-ready, research-oriented predictive maintenance platform for industrial equipment. The project extends the Random Forest failure-prediction workflow from the reference repository by adding comparative machine learning, explainable AI, professional visualization, API deployment, Streamlit dashboarding, and dissertation-style documentation.

Reference baseline: [Yi-Chen-Lin2019/Predictive-maintenance-with-machine-learning](https://github.com/Yi-Chen-Lin2019/Predictive-maintenance-with-machine-learning)

## Why This Project Matters

Traditional maintenance is often reactive or schedule-based. Reactive maintenance waits for breakdowns, while preventive maintenance can replace parts too early. Predictive maintenance uses sensor data to estimate failure risk before downtime happens. This project adds Explainable AI so engineers can understand why a model predicts a fault, for example:

> Machine failure predicted because torque is high, tool wear is excessive, and rotational speed is outside the normal operating range.

That explanation layer is essential in Industry 4.0 environments where maintenance teams need trustworthy, auditable AI support rather than opaque scores.

## Main Capabilities

- Binary machine failure prediction
- Failure type analysis using the AI4I-style feature schema
- Modular preprocessing with missing-value handling, categorical encoding, scaling, and feature engineering
- Comparative models: Random Forest, Logistic Regression, SVM, XGBoost, Gradient Boosting, Decision Tree, optional ANN
- Hyperparameter tuning with cross-validation
- SHAP and LIME explanations for global and local interpretability
- Research-grade plots: heatmaps, feature importance, confusion matrices, SHAP plots, failure distribution, sensor trends
- FastAPI service for deployment
- Streamlit dashboard for upload, prediction, explanation, and monitoring
- MLOps-friendly folder structure for future CI/CD, model monitoring, and cloud deployment

## Dataset

The implementation follows the dataset structure used in the reference project and the common AI4I predictive maintenance schema:

- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Type`
- `Machine failure`
- Failure subtype columns such as `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`

The loader also accepts simplified column names such as `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`, `machine_type`, and `failure_type`.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── dashboard/
├── notebooks/
├── reports/
├── research/
├── src/
│   ├── api/
│   ├── explainability/
│   ├── models/
│   ├── preprocessing/
│   ├── simulation/
│   ├── utils/
│   └── visualization/
├── main.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Models

Use the included sample for a smoke test:

```bash
python main.py train --data data/sample/maintenance_sample.csv --target failure_binary
```

Generate a larger synthetic industrial dataset for more serious testing:

```bash
python -m src.simulation.generate_synthetic_dataset --rows 10000
python main.py train --data data/processed/synthetic_maintenance_large.csv --target failure_binary
```

If your OneDrive folder blocks runtime file creation, choose a writable path:

```powershell
python -m src.simulation.generate_synthetic_dataset --rows 10000 --output "$env:TEMP\synthetic_maintenance_large.csv"
python main.py train --data "$env:TEMP\synthetic_maintenance_large.csv" --target failure_binary
```

Use a full AI4I-style dataset:

```bash
python main.py train --data data/raw/ai4i2020.csv --target failure_binary --tune
```

Training saves model artifacts, metrics, plots, and explainability outputs under `reports/`.

On some OneDrive-synced Windows folders, Python may be unable to create new runtime files inside the project directory. In that case the project automatically falls back to:

```text
%TEMP%\explainable_predictive_maintenance\reports
```

You can choose a different writable output location:

```powershell
$env:EXPLAINABLE_PDM_OUTPUT_DIR="C:\pdm_outputs"
python main.py train --data data/sample/maintenance_sample.csv --target failure_binary
```

## Run the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Important endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

## Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard supports CSV upload, failure-risk scoring, health status viewing, and explanation panels.

## Model Strategy

The Random Forest model is used as the baseline because it is robust, handles nonlinear interactions, and matches the reference repository workflow. The system then compares it with linear, margin-based, boosting, and tree-based alternatives.

| Model | Why included | Industrial trade-off |
| --- | --- | --- |
| Logistic Regression | Transparent baseline | Interpretable but may underfit nonlinear faults |
| Decision Tree | Human-readable rules | Easy to explain but can overfit |
| Random Forest | Strong baseline from reference repo | Robust and practical, less directly interpretable |
| SVM | Good for complex boundaries | Can be slower and harder to explain |
| Gradient Boosting | Strong tabular performance | Higher accuracy, more tuning required |
| XGBoost | Industry-grade boosting | Excellent performance, extra dependency |
| ANN | Optional nonlinear learner | Flexible but requires more data and monitoring |

## Explainable AI

The project integrates:

- SHAP for global and local feature attribution
- LIME for local, human-readable explanations
- Human explanation templates for maintenance teams

Example output:

```text
Predicted status: FAILURE RISK
Reason: High torque, high tool wear, and elevated process temperature increased the failure probability.
Recommended action: Inspect cutting tool condition, check load balance, and schedule maintenance before the next production cycle.
```

## Research Contribution

The research gap addressed here is not simply predicting equipment failure. Many predictive maintenance systems optimize accuracy but do not explain decisions clearly enough for maintenance engineers, auditors, or plant managers. This project contributes an explainable, modular, deployable architecture that connects ML prediction with trustworthy maintenance decision support.

See [research/methodology.md](research/methodology.md), [research/literature_review.md](research/literature_review.md), [reports/architecture.md](reports/architecture.md), [reports/model_card.md](reports/model_card.md), and [reports/deployment_guide.md](reports/deployment_guide.md) for dissertation-ready material.
