import pytest

from src.ml.inference import predict_one


VALID_PAYLOAD = {
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
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


def test_valid_payload_returns_prediction():
    result = predict_one(VALID_PAYLOAD)

    assert "subscribe_probability" in result
    assert "subscribe_label" in result
    assert "threshold_used" in result
    assert "model_name" in result

    assert 0.0 <= result["subscribe_probability"] <= 1.0
    assert result["subscribe_label"] in [0, 1]


def test_invalid_payload_returns_validation_error():
    bad_payload = {
        **VALID_PAYLOAD,
        "job": "influencer",
        "contact": "email",
        "extra_field": "should fail",
    }

    result = predict_one(bad_payload)

    assert result["error"] == "validation_error"
    assert "details" in result

    error_locations = [error["loc"][0] for error in result["details"]]

    assert "job" in error_locations
    assert "contact" in error_locations
    assert "extra_field" in error_locations

    error_types = [error["type"] for error in result["details"]]

    assert "literal_error" in error_types
    assert "extra_forbidden" in error_types