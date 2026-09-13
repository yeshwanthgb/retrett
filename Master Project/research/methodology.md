# Research Methodology

## Title

Explainable Predictive Maintenance System using Machine Learning for Industrial Equipment

## Problem Statement

Industrial equipment failures cause downtime, production loss, safety risk, and increased maintenance cost. Classical maintenance approaches are either reactive, where action is taken after breakdown, or preventive, where maintenance is scheduled at fixed intervals. Both approaches can be inefficient because they do not fully use real-time machine condition data.

This project investigates how machine learning and explainable AI can be combined to predict equipment failure and provide transparent explanations that maintenance teams can trust.

## Research Gap

Many predictive maintenance studies focus on model accuracy but provide limited explanation of why a failure is predicted. In real industrial settings, a maintenance engineer needs to understand the contributing sensor conditions before acting on an AI recommendation. The gap is therefore between high-performing predictive models and explainable, deployable, decision-support systems for Industry 4.0.

## Objectives

1. Build a predictive maintenance system using industrial sensor data.
2. Establish a Random Forest baseline based on the reference repository workflow.
3. Compare multiple machine learning algorithms using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
4. Integrate SHAP and LIME to explain model decisions.
5. Design a deployable architecture with API and dashboard layers.
6. Present results in a format suitable for dissertation, publication, and portfolio demonstration.

## Dataset Understanding

The expected dataset contains operating-condition features including air temperature, process temperature, rotational speed, torque, tool wear, machine type, and failure labels. These features are useful because they represent thermal condition, mechanical load, tool degradation, and machine category.

The project supports both binary failure prediction and failure type classification. Binary prediction answers whether the machine is at risk, while failure type analysis supports targeted maintenance planning.

## Preprocessing Rationale

Missing numerical values are imputed using the median because industrial sensor readings can contain outliers and median imputation is more robust than mean imputation. Categorical machine type values are imputed using the most frequent category and encoded using one-hot encoding. Numerical features are scaled so algorithms such as SVM, Logistic Regression, and ANN are not dominated by high-magnitude variables like rotational speed.

Feature engineering adds:

- Temperature delta: captures thermal stress between process and ambient operating conditions.
- Mechanical power: approximates load using torque and rotational speed.
- Wear-torque ratio: combines degradation and mechanical load.
- Thermal stress index: represents interaction between torque and process heat.

These engineered features are included because equipment failure is often driven by interacting physical conditions rather than isolated sensor values.

## Model Selection

Random Forest is the baseline because it is robust on tabular data, handles nonlinear relationships, and matches the reference project. Logistic Regression provides a transparent statistical baseline. Decision Tree provides rule-like interpretability. SVM tests margin-based nonlinear classification. Gradient Boosting and XGBoost represent strong industry-grade tabular learners. ANN is optional and useful when larger datasets or streaming sensor windows are available.

## Explainability Strategy

SHAP is used for both global and local feature attribution. Global SHAP plots show which features generally influence failure prediction, while local SHAP explanations show how a specific sensor row contributed to one prediction.

LIME is used as a complementary local explanation method. It approximates the trained model around a single instance and produces human-readable feature conditions that influenced the prediction.

Using both methods strengthens the research because it compares explanation consistency across two widely used XAI techniques.

## Evaluation Strategy

Accuracy alone can be misleading because failure events are usually rare. Precision measures how many predicted failures are actually failures. Recall measures how many actual failures are detected. F1 score balances precision and recall. ROC-AUC evaluates ranking quality. Confusion matrices reveal false positives and false negatives, which are especially important in maintenance planning.

False negatives are costly because missed failures can cause downtime. False positives are also costly because unnecessary maintenance wastes resources. The best industrial model is therefore not always the model with the highest accuracy; it is the model with the best operational trade-off.

## Industry Relevance

The system supports Industry 4.0 by connecting sensor-driven monitoring, machine learning, explainability, API deployment, and dashboard-based decision support. It can be extended with IoT ingestion, model monitoring, alerting, cloud deployment, and Remaining Useful Life estimation.

