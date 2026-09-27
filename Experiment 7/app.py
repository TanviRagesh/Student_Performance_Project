
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/student-mat.csv"

FEATURES = ["Hours_Studied", "Attendance", "Previous_Scores"]
TARGET = "Exam_Score"


def load_data(path=DATA_PATH):
    """Load and validate the student dataset."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if len(df) < 2:
        raise ValueError("At least two valid student records are required.")

    return df


def train_model(df):
    """Train the student exam score prediction model."""
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )
    model.fit(df[FEATURES], df[TARGET])
    return model


def predict_score(model, hours, attendance, previous_scores):
    """Predict an exam score for a student."""
    if hours < 0 or not 0 <= attendance <= 100:
        raise ValueError("Invalid study hours or attendance.")

    if not 0 <= previous_scores <= 100:
        raise ValueError("Previous scores must be between 0 and 100.")

    student = pd.DataFrame(
        [[hours, attendance, previous_scores]],
        columns=FEATURES
    )
    prediction = model.predict(student)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)

    score = predict_score(model, 25, 85, 75)
    print(f"Predicted Exam Score: {score}")
