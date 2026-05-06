"""Tests for prediction endpoint: valid inputs return predictions; invalid inputs return 422."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _mock_pipeline_predict_proba(proba: float):
    """Return a mock pipeline that always predicts the given probability."""
    pipeline = MagicMock()
    pipeline.predict_proba = MagicMock(return_value=[[1 - proba, proba]])
    return pipeline


@pytest.fixture()
def client():
    """Create a test client with mocked lifespan singletons."""
    from ml.reference_stats import ReferenceStats

    ref_stats = ReferenceStats(
        numeric={
            "euribor3m": {
                "mean": 3.0,
                "std": 1.5,
                "quantiles": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                "reference_values": [3.0] * 100,
            },
        },
        categorical={"job": {"admin.": 0.25, "blue-collar": 0.22}},
        output_proportions={"0": 0.89, "1": 0.11},
        dataset_hash="test_hash",
    )

    with patch("service.main.get_settings") as mock_settings, \
         patch("service.main.load_model") as mock_load, \
         patch("service.main.load_reference_stats", return_value=ref_stats), \
         patch("service.main.create_async_engine"), \
         patch("service.main.async_sessionmaker"):
        from core.settings import Settings

        mock_settings.return_value = Settings(
            google_api_key="test-key-for-testing",
            postgres_password="testpassword",
            promotion_api_key="test_promotion_key_16ch",
        )
        mock_load.return_value = (_mock_pipeline_predict_proba(0.75), 0.5)

        from service.main import app

        with TestClient(app) as c:
            yield c


def test_predict_valid_input_returns_200(client):
    payload = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.9,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.3,
        "nr.employed": 5099.1,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "label" in data
    assert "probability" in data
    assert "threshold" in data


def test_predict_invalid_age_returns_422(client):
    payload = {
        "age": 10,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.9,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.3,
        "nr.employed": 5099.1,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    payload = {
        "age": 35,
        "job": "admin.",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422