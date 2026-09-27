
# Responsible AI Report
## Experiment 8: Student Performance Dashboard

## 1. Introduction

This project develops an educational student performance
prediction dashboard using Streamlit and a Random Forest
regression model.

The application predicts student exam scores using
academic and other student-related features. It also
provides model evaluation, SHAP explanations and basic
data drift monitoring.

Responsible AI principles are considered throughout the
development of the application.

## 2. Fairness

Fairness is important because a prediction model can
produce different errors for different student groups.

Measures implemented in this project:

- Direct demographic fields such as sex and gender are
  excluded from model training.
- Demographic fields are retained separately for
  exploratory fairness auditing.
- Group-level Mean Absolute Error (MAE) can be compared
  for available demographic groups.
- SHAP explanations are used to inspect model behavior.

Limitations:

- Excluding sensitive attributes does not eliminate
  bias from correlated proxy features.
- Differences in group MAE do not, by themselves,
  establish discrimination.
- Small or imbalanced groups may produce unreliable
  fairness estimates.
- The audit does not establish that the model is fair.

Future improvements:

- Conduct a more comprehensive fairness assessment.
- Examine proxy variables and dataset representation.
- Evaluate performance across different student groups.
- Have domain experts review the audit.

## 3. Privacy

Student datasets can contain sensitive personal
information. Privacy must be considered during
development and deployment.

Measures:

- The dashboard does not require students' names.
- Student_ID is excluded from model training.
- Only the features needed for prediction are used.
- Dataset files should be reviewed before publication.
- Identifiable records should not be exposed in a
  public GitHub repository.

Limitations:

- The prototype does not implement authentication
  or role-based access control.
- Uploaded files are processed by the application.
- Public deployment should use only appropriately
  anonymized or synthetic data.

## 4. Consent

Before collecting or using real student data,
appropriate authorization and consent should be
obtained according to the applicable institutional
requirements.

This project uses an educational dataset for
demonstration purposes.

If the application is adapted to use real student
records, its developers should:

- Explain the purpose of data collection.
- Obtain the required permissions.
- Limit data collection to what is necessary.
- Explain how data will be used and retained.
- Respect applicable institutional and legal
  requirements.

## 5. Transparency and Explainability

The dashboard displays model performance metrics,
predictions and SHAP-based feature explanations.

The following are provided:

- MAE to show average absolute prediction error.
- RMSE to give greater weight to larger errors.
- R² to describe the proportion of target variation
  explained by the model on the test set.
- SHAP feature importance to summarize model behavior.

SHAP values explain the model's learned relationships.
They do not establish causal relationships.

Predictions should be treated as estimates rather
than guaranteed outcomes.

## 6. Model Reliability

The model is evaluated using an 80:20 random
train-test split.

The application reports MAE, RMSE and R² on the
held-out test data.

These metrics are specific to the dataset and split
used in the experiment. They do not guarantee
performance on future students or other schools.

Further validation on independent datasets is
recommended before any real-world use.

## 7. Data Drift

The application provides basic data drift checks.

Numerical features are evaluated using the
Population Stability Index (PSI).

Categorical features are evaluated using
Total Variation Distance (TVD).

The dashboard uses illustrative thresholds to
classify drift as low, moderate or high.

Drift indicates a change in feature distributions.
It does not automatically indicate a decline in
prediction accuracy.

Actual model performance should be evaluated
using appropriate ground-truth outcomes.

## 8. Human Oversight

The dashboard is intended for educational and
demonstration purposes.

Its predictions must not be used as the sole basis
for decisions affecting students.

Teachers and other authorized human reviewers
should consider additional context and the
limitations of the model.

The application does not replace professional
judgment or institutional assessment.

## 9. Security and Deployment

The application is designed as a prototype.

Before deployment:

- Review all data files for sensitive information.
- Do not commit passwords, API keys or secrets.
- Restrict access to confidential datasets.
- Use anonymized or synthetic data for public demos.
- Review the hosting platform's privacy and
  data-handling policies.

## 10. Known Limitations

1. Random Forest predictions may not generalize
   to different schools or student populations.
2. The model may inherit biases from its training data.
3. SHAP explanations describe model behavior,
   not causation.
4. The drift checks are basic distribution comparisons.
5. The application does not provide a full privacy
   or fairness certification.
6. The prototype has no authentication system.
7. Prediction accuracy depends on data quality
   and the relevance of the input features.

## 11. Future Improvements

- Test on independent student datasets.
- Add comprehensive fairness metrics.
- Introduce authentication and access controls.
- Improve data validation and monitoring.
- Add model versioning and retraining.
- Conduct a formal privacy and security review.
- Collect feedback from relevant educational experts.

## 12. Conclusion

This experiment demonstrates the integration of
machine learning, interactive visualization,
explainable AI and basic responsible AI practices.

Fairness, privacy, consent, transparency and human
oversight are important considerations when
developing student performance prediction systems.

The application is an educational prototype and
requires further validation before any real-world
deployment.