"""Tests for prediction feature engineering and batch inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml.inference import predict_batch, prepare_prediction_features
from ml.schema import BankMarketingRequest

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


def test_valid_payload_parses():
    request = BankMarketingRequest.model_validate(VALID_PAYLOAD)
    assert request.age == 35
    assert request.pdays == 999
    assert request.emp_var_rate == 1.1


def test_invalid_payload_rejects_unknown_job():
    bad_payload = {
        **VALID_PAYLOAD,
        "job": "influencer",
    }
    with pytest.raises(Exception):
        BankMarketingRequest.model_validate(bad_payload)


def test_extra_fields_rejected():
    bad_payload = {
        **VALID_PAYLOAD,
        "extra_field": "should fail",
    }
    with pytest.raises(Exception):
        BankMarketingRequest.model_validate(bad_payload)


def test_prepare_prediction_features_adds_pdays_was_999():
    feature_dict = dict(VALID_PAYLOAD)
    result = prepare_prediction_features(feature_dict)
    assert "pdays_was_999" in result
    assert result["pdays_was_999"] == 1


def test_prepare_prediction_features_flags_non_999_pdays():
    payload = {**VALID_PAYLOAD, "pdays": 5}
    feature_dict = dict(payload)
    result = prepare_prediction_features(feature_dict)
    assert result["pdays_was_999"] == 0


class TestPredictBatch:
    def test_predict_batch_returns_probabilities(self):
        X = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, [0, 0, 1])
        result = predict_batch(X, pipeline)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert all(0 <= p <= 1 for p in result)

    def test_predict_batch_positive_class(self):
        X = pd.DataFrame({"feat": [1.0, 2.0]})
        pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, [0, 1])
        result = predict_batch(X, pipeline)
        assert result.shape == (2,)
