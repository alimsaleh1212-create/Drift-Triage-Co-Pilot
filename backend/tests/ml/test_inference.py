"""Tests for prediction validation and inference."""

from __future__ import annotations

import pytest

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


def test_pdays_was_999_added():
    from ml.inference import prepare_input_row

    request = BankMarketingRequest.model_validate(VALID_PAYLOAD)
    row = prepare_input_row(request)
    assert "pdays_was_999" in row.columns
    assert row["pdays_was_999"].iloc[0] == 1


def test_pdays_not_999_flag():
    from ml.inference import prepare_input_row

    payload = {**VALID_PAYLOAD, "pdays": 5}
    request = BankMarketingRequest.model_validate(payload)
    row = prepare_input_row(request)
    assert row["pdays_was_999"].iloc[0] == 0