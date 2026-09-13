# Model Card

## Intended Use

The model predicts industrial machine failure risk from sensor and machine-condition data. It is intended for maintenance decision support, inspection prioritization, dashboard monitoring, and research demonstration.

It should not be used as the only authority for safety-critical shutdown decisions without plant-specific validation, human review, and integration with engineering rules.

## Inputs

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Engineered physical-condition features

## Outputs

- Binary failure prediction
- Optional failure type prediction
- Prediction confidence when the selected model supports probabilities
- Human-readable explanation
- SHAP and LIME explanation artifacts

## Evaluation Metrics

The main model-selection metrics are F1 score and ROC-AUC. Accuracy is reported but not used alone because failure datasets are often imbalanced.

Operational interpretation:

- False negative: failure missed; high downtime and safety risk.
- False positive: unnecessary inspection; lower risk but higher maintenance cost.
- High recall is preferred when downtime risk is expensive.
- High precision is preferred when maintenance capacity is constrained.

## Limitations

The sample dataset is only for smoke testing. Dissertation-quality results require the full AI4I-style dataset or real plant sensor data. Model explanations describe statistical evidence, not guaranteed physical causality. Production deployment should include drift detection, calibration checks, periodic retraining, and maintenance outcome feedback.

## Ethical and Operational Considerations

Explainability should support human judgment rather than replace it. Maintenance workers should be trained to interpret explanations and challenge predictions that conflict with physical inspection evidence.
