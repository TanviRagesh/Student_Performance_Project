
# Experiment 8: Dashboard, Responsible AI Reporting
# and Final Portfolio

## Aim

To develop an interactive dashboard for student
performance predictions, model explainability,
data drift monitoring and responsible AI reporting.

## Objective

- Develop a Streamlit dashboard.
- Predict student exam scores.
- Visualize model performance metrics.
- Explain predictions using SHAP.
- Perform basic data drift checks.
- Document fairness, privacy and consent.
- Publish the final application and report on GitHub.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- SHAP
- Matplotlib
- Git and GitHub

## Project Structure

```text
Experiment 8/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- Responsible_AI.md
|-- .gitignore
|
|-- data/
|   |-- student_performance.csv
|
|-- notebooks/
|
|-- reports/
```

## Features

### 1. Dataset Overview
- Dataset preview
- Record and feature counts
- Missing-value summary

### 2. Model Evaluation
- Random Forest regression
- 80:20 train-test split
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared score

### 3. Predictions
- Individual student score prediction
- Batch predictions from uploaded CSV files
- Downloadable prediction results

### 4. Explainable AI
- SHAP TreeExplainer
- Global feature importance
- Visual explanation of model behavior

### 5. Data Drift
- Population Stability Index (PSI)
  for numerical features
- Total Variation Distance (TVD)
  for categorical features
- Downloadable drift report

### 6. Responsible AI
- Demographic group error comparisons
- Fairness checklist
- Privacy and consent considerations
- Model limitations and human oversight

## Dataset

The application uses a student performance dataset
with `Exam_Score` as the prediction target.

Place the full dataset at:

`data/student_performance.csv`

Alternatively, upload a compatible CSV through
the dashboard sidebar.

The model excludes Student_ID, sex and Gender
from its training features. Demographic attributes
may be retained for exploratory fairness auditing.

## Installation

Create and activate a Python virtual environment.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

From the Experiment 8 directory, run:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal,
usually http://localhost:8501.

## Responsible AI

Refer to `Responsible_AI.md` for the fairness,
privacy, consent, transparency and limitations
documentation.

## Limitations

This project is an educational prototype.
Predictions are estimates, and the model has not
been certified for real-world educational decisions.

## Deployment

The application can be deployed using
Streamlit Community Cloud.

Select the GitHub repository, the main branch
and `Experiment 8/app.py` as the application
entry point.

## Conclusion

The experiment demonstrates an interactive
machine learning dashboard that integrates
predictions, model evaluation, explainability,
data drift monitoring and responsible AI reporting.