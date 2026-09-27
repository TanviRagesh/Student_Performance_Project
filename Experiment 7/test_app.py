
import pytest
from app import load_data, train_model, predict_score


def test_dataset_loads():
    df = load_data()
    assert not df.empty
    assert {"Hours_Studied", "Attendance",
            "Previous_Scores", "Exam_Score"}.issubset(df.columns)


def test_model_trains():
    df = load_data()
    model = train_model(df)
    assert model is not None


def test_prediction_is_numeric():
    df = load_data()
    model = train_model(df)

    score = predict_score(model, 20, 80, 70)

    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_invalid_attendance():
    df = load_data()
    model = train_model(df)

    with pytest.raises(ValueError):
        predict_score(model, 20, 120, 70)


def test_invalid_study_hours():
    df = load_data()
    model = train_model(df)

    with pytest.raises(ValueError):
        predict_score(model, -1, 80, 70)
