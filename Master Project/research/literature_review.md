# Literature Review

## Predictive Maintenance

Predictive maintenance uses operational data to estimate future failure risk and recommend maintenance before breakdown. Compared with reactive maintenance, it reduces unplanned downtime. Compared with fixed preventive maintenance, it can reduce unnecessary part replacement and labor cost.

Carvalho et al. describe predictive maintenance as an important application area for machine learning because industrial systems generate large volumes of sensor and maintenance data. Their systematic review shows that classification, regression, anomaly detection, and remaining-useful-life estimation are common predictive maintenance tasks.

## Industry 4.0 and Smart Manufacturing

Industry 4.0 connects machines, sensors, edge devices, cloud systems, and analytics platforms. In this setting, predictive maintenance becomes a decision-support layer: sensor data is collected continuously, machine learning models estimate risk, and maintenance systems trigger inspection or work orders.

The practical challenge is that industrial users need more than a prediction score. A plant engineer needs to know whether the model is reacting to high torque, thermal stress, tool wear, abnormal speed, or another operating condition.

## Machine Learning for Failure Prediction

Tree-based models are widely used for tabular predictive maintenance because they handle nonlinear relationships and feature interactions. Random Forest is a strong baseline because it reduces single-tree overfitting through ensemble averaging. Gradient Boosting and XGBoost often improve performance by sequentially learning from previous errors. Logistic Regression and Decision Tree models remain important because they provide interpretable baselines.

In this project, model comparison is treated as a research task rather than a coding formality. Each model is evaluated with operationally relevant metrics because failure data is commonly imbalanced.

## Explainable AI

SHAP, introduced by Lundberg and Lee, provides feature attributions based on Shapley values. This helps answer which variables increased or decreased a prediction. SHAP is useful for both global feature ranking and local prediction-level explanation.

LIME, introduced by Ribeiro, Singh, and Guestrin, explains individual predictions by fitting a local interpretable model around the instance being explained. LIME is model-agnostic, so it can explain Random Forest, SVM, XGBoost, and neural network predictions.

This project uses both SHAP and LIME because explanation consistency is valuable in industrial AI. If two explanation methods highlight torque and tool wear as important for the same fault, the maintenance team can have greater confidence in the recommendation.

## Research Gap Addressed

The main research gap is the gap between accurate predictive maintenance models and deployable, explainable maintenance decision support. Many projects stop after training a classifier. This project extends the workflow into a full platform with preprocessing, model comparison, explanation, visualization, API serving, dashboarding, and deployment architecture.

## Key Sources

- AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions: https://arxiv.org/abs/1705.07874
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier: https://arxiv.org/abs/1602.04938
- Carvalho, T. P. et al. (2019). A systematic literature review of machine learning methods applied to predictive maintenance. Computers & Industrial Engineering, 137, 106024.
