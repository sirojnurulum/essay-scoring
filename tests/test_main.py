import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from app.main import app, startup_event

# Override the startup event to prevent original models from loading during tests
async def mock_startup():
    print("Mock startup event executed")

app.dependency_overrides[startup_event] = mock_startup

# Create a TestClient instance after overriding dependencies
client = TestClient(app)


class MockFeatureEngineer:
    def transform(self, student_df, reference_df):
        return np.array([[0.85, 0.9, 1.1]])

class MockModel:
    def predict(self, features):
        return np.array([9.5])


@pytest.fixture
def mock_model_and_db(monkeypatch):
    """Pytest fixture to mock model loading and database connections."""
    # 1. Mock the model dictionary
    mock_models_dict = {
        ('matematika', 'x'): {
            'feature_engineer': MockFeatureEngineer(),
            'model': MockModel()
        }
    }
    monkeypatch.setattr("app.main.models", mock_models_dict)

    # 2. Mock the database engine and call
    def mock_read_sql(*args, **kwargs):
        return pd.DataFrame({'jawaban_referensi_terbaik': ["dummy answer"]})

    monkeypatch.setattr(pd, "read_sql", mock_read_sql)
    # Also mock the engine to pass the `if not engine` check
    monkeypatch.setattr("app.main.engine", True)


def test_predict_score_success(mock_model_and_db):
    """
    Tests the /score endpoint for a successful request.
    It verifies that the endpoint returns a 200 OK status and the expected score.
    """
    request_payload = {
        "subject": "Matematika",
        "grade_level": "X",
        "question_id": 1,
        "answer_text": "Ini adalah jawaban siswa."
    }
    response = client.post("/score", json=request_payload)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    response_data = response.json()
    assert "score" in response_data
    assert response_data["score"] == 9.5


def test_predict_score_model_not_found(mock_model_and_db):
    """
    Tests the /score endpoint for a case where the requested model does not exist.
    It verifies that the endpoint returns a 404 Not Found status.
    """
    request_payload = {
        "subject": "Fisika",
        "grade_level": "XI",
        "question_id": 2,
        "answer_text": "Ini adalah jawaban lain."
    }
    response = client.post("/score", json=request_payload)
    assert response.status_code == 404, f"Expected status code 404, but got {response.status_code}"
    assert "No model found" in response.json()["detail"]
