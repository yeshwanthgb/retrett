# System Architecture

## Layered Architecture

```mermaid
flowchart LR
    A["Industrial Sensor Data"] --> B["Data Layer"]
    B --> C["Preprocessing Layer"]
    C --> D["ML Prediction Engine"]
    D --> E["Explainability Layer"]
    E --> F["API Layer"]
    F --> G["Dashboard / UI Layer"]
    F --> H["Alert System"]
    D --> I["Model Monitoring"]
```

## Data Layer

The data layer stores raw sensor records, processed datasets, and metadata. In production, this layer can connect to IoT gateways, historians, PLC systems, message queues, or cloud object storage.

## Preprocessing Layer

This layer validates columns, handles missing values, encodes machine type, scales numerical features, and creates engineered features. The same preprocessing pipeline is used during training and inference to prevent training-serving skew.

## ML Prediction Engine

The prediction engine trains and compares multiple models. The best model is selected using F1 score and ROC-AUC because maintenance datasets are often imbalanced. The trained pipeline is serialized as a single artifact so preprocessing and prediction remain consistent.

## Explainability Layer

SHAP provides global and local feature attribution. LIME provides local explanations that are easy to translate into maintenance language. Human-readable explanation templates convert model evidence into actionable engineering statements.

## API Layer

FastAPI exposes health, single-prediction, and batch-prediction endpoints. This enables integration with dashboards, plant systems, and alerting services.

## Dashboard Layer

The Streamlit dashboard supports CSV upload, simulated sensor data, failure-risk scoring, KPI monitoring, local explanation, and sensor trend visualization.

## Alert System

An alert service can be added to send email, SMS, Slack, Microsoft Teams, or maintenance-ticket notifications when predicted risk crosses a threshold.

## Deployment View

```mermaid
flowchart TB
    S["Sensors / IoT Gateway"] --> Q["Streaming or Batch Ingestion"]
    Q --> P["Feature Pipeline"]
    P --> M["Model Service"]
    M --> X["SHAP / LIME Explanation Service"]
    X --> API["FastAPI"]
    API --> UI["Streamlit or React Dashboard"]
    API --> Alert["Alert Manager"]
    API --> Store["Prediction Logs and Monitoring Database"]
```

## MLOps Extension

For an industry deployment, add data validation, experiment tracking, model registry, drift monitoring, automated retraining, and CI/CD. Monitoring should track input drift, prediction drift, latency, explanation stability, and maintenance outcome feedback.

