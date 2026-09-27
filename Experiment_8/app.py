
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Performance AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Student Performance AI Dashboard")
st.caption(
    "Experiment 8 | Predictions, Explainable AI, "
    "Model Evaluation and Responsible AI"
)


# ---------------------------------------------------------
# 2. FIND THE DATASET
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent

DATA_PATHS = [
    APP_DIR / "data" / "student_performance.csv",
    REPO_DIR / "data" / "processed" /
    "Cleaned_Student_Performance_Dataset.csv",
    REPO_DIR / "data" / "processed" /
    "StudentPerformance.csv",
    REPO_DIR / "Experiment 7" / "data" /
    "student-mat.csv",
]


def find_dataset():
    """Find an existing dataset in the repository."""
    for path in DATA_PATHS:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes):
    """Load a CSV file and clean column names."""
    from io import BytesIO

    data = pd.read_csv(BytesIO(file_bytes))
    data.columns = data.columns.astype(str).str.strip()
    return data


uploaded_file = st.sidebar.file_uploader(
    "Upload student dataset (CSV)",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file.getvalue())
        st.sidebar.success("Uploaded dataset loaded.")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    default_path = find_dataset()

    if default_path is None:
        st.warning(
            "No dataset was found automatically. "
            "Upload your complete student dataset using "
            "the sidebar."
        )
        st.stop()

    try:
        df = load_dataset(default_path.read_bytes())
        st.sidebar.success(f"Dataset: {default_path.name}")
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()


TARGET = "Exam_Score"

if TARGET not in df.columns:
    st.error(
        f"Target column '{TARGET}' was not found. "
        "Check that you uploaded the correct dataset."
    )
    st.stop()

# Keep a copy of the original data for auditing.
df = df.copy()

# Remove rows with a missing or invalid target.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

if len(df) < 10:
    st.error(
        "At least 10 valid rows are required for this "
        "demonstration. Upload your complete dataset."
    )
    st.stop()


# ---------------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# ---------------------------------------------------------

# These fields are excluded from model training.
# Demographic fields are retained in df for auditing.
EXCLUDED_COLUMNS = {
    "exam_score",
    "student_id",
    "id",
    "sex",
    "gender",
}

feature_columns = [
    col for col in df.columns
    if col.strip().lower() not in EXCLUDED_COLUMNS
]

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Remove columns that contain no usable values.
usable_columns = X.columns[X.notna().any()].tolist()
X = X[usable_columns]
feature_columns = usable_columns

if X.shape[1] == 0:
    st.error("No usable model features were found.")
    st.stop()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# Avoid an invalid empty transformer.
transformers = []

if numeric_features:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    transformers.append(
        ("categorical", categorical_pipeline,
         categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)


# ---------------------------------------------------------
# 5. TRAIN THE RANDOM FOREST MODEL
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Training the model...")
def train_model(data, target):
    """Train and evaluate a Random Forest regression model."""

    X_data = data[feature_columns]
    y_data = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y_data,
        test_size=0.20,
        random_state=42,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions),
    }

    return (
        model_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        metrics,
    )


try:
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        metrics,
    ) = train_model(df, TARGET)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()


# ---------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Dashboard Navigation")

st.sidebar.info(
    "This dashboard uses a Random Forest regression "
    "model to predict student exam scores."
)

st.sidebar.caption(
    "Training/test split: 80/20 | Random state: 42"
)

st.sidebar.caption(
    "Sensitive demographic fields are excluded "
    "from the model and retained for auditing."
)


# ---------------------------------------------------------
# 7. DASHBOARD TABS
# ---------------------------------------------------------

tab_overview, tab_predict, tab_shap, tab_drift, tab_ai = (
    st.tabs(
        [
            "📊 Overview",
            "🎯 Predictions",
            "🔍 SHAP Explainability",
            "📈 Drift Checks",
            "🛡️ Responsible AI",
        ]
    )
)


# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------

with tab_overview:
    st.header("Dataset and Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Model Features", len(feature_columns))
    col3.metric("Training Records", len(X_train))
    col4.metric("Testing Records", len(X_test))

    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric("MAE", f"{metrics['MAE']:.3f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    m3.metric("R² Score", f"{metrics['R2']:.3f}")

    st.caption(
        "Metrics are calculated on the held-out test set. "
        "They measure performance on this random split, "
        "not on future students."
    )

    st.subheader("Actual vs Predicted Scores")

    comparison = pd.DataFrame(
        {
            "Actual Score": y_test.to_numpy(),
            "Predicted Score": y_pred,
        }
    ).reset_index(drop=True)

    st.dataframe(comparison.head(20), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        comparison["Actual Score"],
        comparison["Predicted Score"],
        alpha=0.7,
    )

    minimum = min(
        comparison["Actual Score"].min(),
        comparison["Predicted Score"].min(),
    )
    maximum = max(
        comparison["Actual Score"].max(),
        comparison["Predicted Score"].max(),
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    ax.set_xlabel("Actual Exam Score")
    ax.set_ylabel("Predicted Exam Score")
    ax.set_title("Actual vs Predicted Exam Scores")
    ax.grid(alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Missing Values")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found in the dataset.")
    else:
        missing_df = missing.rename(
            "Missing Count"
        ).reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        st.dataframe(missing_df, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: PREDICTIONS
# ---------------------------------------------------------

with tab_predict:
    st.header("Student Exam Score Prediction")

    st.write(
        "Choose a record from the dataset to see its "
        "predicted exam score."
    )

    selected_row = st.number_input(
        "Select student record (row number)",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1,
    )

    row_index = int(selected_row) - 1
    student_features = X.iloc[[row_index]]

    with st.expander("View selected student's input data"):
        st.dataframe(
            student_features,
            use_container_width=True,
        )

    if st.button("Predict Exam Score", type="primary"):
        prediction = model.predict(student_features)[0]

        st.success("Prediction generated successfully.")

        st.metric(
            "Predicted Exam Score",
            f"{prediction:.2f}",
        )

        st.caption(
            "This is a model estimate, not a guaranteed "
            "exam result."
        )

    st.divider()

    st.subheader("Batch Prediction")

    st.write(
        "Upload a CSV containing new student records. "
        "The target column is optional."
    )

    batch_file = st.file_uploader(
        "Upload records for prediction",
        type=["csv"],
        key="batch_upload",
    )

    if batch_file is not None:
        try:
            new_data = load_dataset(batch_file.getvalue())

            missing_features = [
                col for col in feature_columns
                if col not in new_data.columns
            ]

            if missing_features:
                st.error(
                    "Missing input columns: "
                    + ", ".join(missing_features)
                )
            else:
                batch_X = new_data[feature_columns]
                batch_predictions = model.predict(batch_X)

                results = new_data.copy()
                results["Predicted_Exam_Score"] = (
                    batch_predictions
                )

                st.subheader("Batch Prediction Results")
                st.dataframe(
                    results.head(50),
                    use_container_width=True,
                )

                csv_data = results.to_csv(index=False).encode(
                    "utf-8"
                )

                st.download_button(
                    "Download predictions as CSV",
                    data=csv_data,
                    file_name="student_predictions.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


# ---------------------------------------------------------
# TAB 3: SHAP EXPLAINABILITY
# ---------------------------------------------------------

with tab_shap:
    st.header("SHAP Model Explainability")

    st.write(
        "SHAP estimates how individual input features "
        "contribute to the model's predictions. The bar "
        "chart below summarizes the average absolute "
        "contribution across a sample of test records."
    )

    if st.button("Generate SHAP Explanation"):
        try:
            fitted_preprocessor = model.named_steps[
                "preprocessor"
            ]
            fitted_rf = model.named_steps["model"]

            sample = X_test.iloc[
                :min(100, len(X_test))
            ]

            transformed = fitted_preprocessor.transform(sample)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            transformed = np.asarray(transformed)

            feature_names = (
                fitted_preprocessor.get_feature_names_out()
            )

            explainer = shap.TreeExplainer(fitted_rf)
            shap_values = explainer.shap_values(transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]

            if shap_values.ndim != 2:
                raise ValueError(
                    "Unexpected SHAP output dimensions."
                )

            importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Mean Absolute SHAP": importance,
                }
            ).sort_values(
                "Mean Absolute SHAP",
                ascending=False,
            ).head(15)

            st.subheader("Most Influential Features")

            fig, ax = plt.subplots(figsize=(10, 6))

            plot_data = importance_df.sort_values(
                "Mean Absolute SHAP",
                ascending=True,
            )

            ax.barh(
                plot_data["Feature"],
                plot_data["Mean Absolute SHAP"],
            )

            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_ylabel("Feature")
            ax.set_title("Top 15 SHAP Feature Importances")
            ax.grid(axis="x", alpha=0.25)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                importance_df,
                use_container_width=True,
            )

            st.caption(
                "SHAP importance describes model behavior, "
                "not causation. One-hot encoded categories "
                "appear as separate features."
            )

        except Exception as exc:
            st.error(f"SHAP calculation failed: {exc}")


# ---------------------------------------------------------
# TAB 4: DATA DRIFT
# ---------------------------------------------------------

def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index for numeric data."""

    reference = pd.to_numeric(
        reference, errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current, errors="coerce"
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    if reference.nunique() < 2:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    internal_edges = np.unique(
        reference.quantile(quantiles).to_numpy()
    )

    edges = np.concatenate(
        ([-np.inf], internal_edges, [np.inf])
    )

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = np.maximum(
        ref_counts / len(reference), 0.0001
    )
    cur_pct = np.maximum(
        cur_counts / len(current), 0.0001
    )

    return float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )
    )


def calculate_tvd(reference, current):
    """Calculate total variation distance for categories."""

    reference = reference.fillna("Missing").astype(str)
    current = current.fillna("Missing").astype(str)

    categories = set(reference.unique()) | set(current.unique())

    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)

    difference = sum(
        abs(
            ref_dist.get(category, 0)
            - cur_dist.get(category, 0)
        )
        for category in categories
    )

    return float(difference / 2)


with tab_drift:
    st.header("Data Drift Monitoring")

    st.write(
        "Compare the training reference data with a new "
        "dataset to identify changes in feature distributions."
    )

    st.info(
        "Drift is a change in the input-data distribution. "
        "It does not, by itself, prove that model accuracy "
        "has decreased."
    )

    drift_file = st.file_uploader(
        "Upload a new/current dataset (CSV)",
        type=["csv"],
        key="drift_upload",
    )

    if drift_file is None:
        st.warning(
            "Upload a separate current dataset to run "
            "the drift analysis. The training data will "
            "be used as the reference."
        )
    else:
        try:
            current_df = load_dataset(drift_file.getvalue())

            current_df.columns = (
                current_df.columns.astype(str).str.strip()
            )

            drift_results = []

            for column in feature_columns:
                if column not in current_df.columns:
                    continue

                reference_values = X_train[column]
                current_values = current_df[column]

                if column in numeric_features:
                    score = calculate_psi(
                        reference_values,
                        current_values,
                    )
                    method = "PSI"

                    if pd.isna(score):
                        status = "Insufficient variation"
                    elif score < 0.10:
                        status = "Low"
                    elif score < 0.20:
                        status = "Moderate"
                    else:
                        status = "High"

                else:
                    score = calculate_tvd(
                        reference_values,
                        current_values,
                    )
                    method = "TVD"

                    if score < 0.05:
                        status = "Low"
                    elif score < 0.15:
                        status = "Moderate"
                    else:
                        status = "High"

                drift_results.append(
                    {
                        "Feature": column,
                        "Method": method,
                        "Drift Score": score,
                        "Status": status,
                    }
                )

            if drift_results:
                drift_df = pd.DataFrame(drift_results)

                st.subheader("Drift Results")
                st.dataframe(
                    drift_df,
                    use_container_width=True,
                )

                st.download_button(
                    "Download drift report",
                    data=drift_df.to_csv(index=False).encode(
                        "utf-8"
                    ),
                    file_name="data_drift_report.csv",
                    mime="text/csv",
                )

                st.caption(
                    "PSI and TVD thresholds are illustrative "
                    "screening thresholds, not universal "
                    "statistical significance tests."
                )
            else:
                st.warning(
                    "No overlapping model features were found."
                )

        except Exception as exc:
            st.error(f"Drift analysis failed: {exc}")


# ---------------------------------------------------------
# TAB 5: RESPONSIBLE AI
# ---------------------------------------------------------

with tab_ai:
    st.header("Responsible AI and Fairness")

    st.write(
        "This section provides a basic fairness audit "
        "using the held-out test records."
    )

    audit_columns = [
        col for col in df.columns
        if col.lower() in {"sex", "gender"}
    ]

    if not audit_columns:
        st.warning(
            "No sex or gender column is available "
            "for a group fairness audit."
        )
    else:
        selected_group = st.selectbox(
            "Choose a demographic audit field",
            audit_columns,
        )

        test_groups = df.loc[
            X_test.index, selected_group
        ].reset_index(drop=True)

        actual = y_test.reset_index(drop=True)
        predicted = pd.Series(y_pred)

        fairness_data = pd.DataFrame(
            {
                "Group": test_groups.fillna("Missing").astype(str),
                "Actual": actual,
                "Predicted": predicted,
            }
        )

        fairness_data["Absolute Error"] = (
            fairness_data["Actual"]
            - fairness_data["Predicted"]
        ).abs()

        fairness_summary = (
            fairness_data.groupby("Group", dropna=False)
            .agg(
                Sample_Count=("Absolute Error", "size"),
                MAE=("Absolute Error", "mean"),
                Mean_Prediction=("Predicted", "mean"),
                Mean_Actual=("Actual", "mean"),
            )
            .reset_index()
        )

        st.subheader("Group-Level Performance")

        st.dataframe(
            fairness_summary.round(3),
            use_container_width=True,
        )

        st.caption(
            "Group MAE differences are descriptive. Small "
            "groups and differences in sample composition "
            "can make comparisons unstable. This is not "
            "a complete fairness certification."
        )

    st.subheader("Responsible AI Checklist")

    checklist = [
        "Check model performance on held-out data.",
        "Inspect errors across demographic groups.",
        "Exclude direct demographic attributes from training.",
        "Check for proxy variables and historical bias.",
        "Collect only the data necessary for the task.",
        "Obtain appropriate consent before collecting personal data.",
        "Avoid publishing identifiable student records.",
        "Explain predictions and their limitations.",
        "Provide human review instead of relying solely on predictions.",
        "Monitor incoming data for distribution changes.",
    ]

    for item in checklist:
        st.checkbox(item, key=f"rai_{item}")

    st.warning(
        "This is an educational prototype. It should not "
        "be used to make high-stakes decisions about students."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Experiment 8 | Student Performance Dashboard | "
    "Educational demonstration"
)