"""Pydantic schemas for prediction request and response.

Per CLAUDE.md §10, every endpoint has a Pydantic model for its request
body AND another for its response. The request model validates incoming
prediction data using ``extra="forbid"`` so structurally invalid inputs
are rejected before reaching any logic.

Column names that contain dots (e.g. ``emp.var.rate``) are accepted
via aliases so the JSON payload matches the original dataset exactly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BankMarketingRequest(BaseModel):
    """Validated single prediction request for the Bank Marketing model.

    Columns match the model's expected features after dropping ``duration``.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    age: int = Field(ge=17, le=100)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y", "basic.6y", "basic.9y", "high.school",
        "illiterate", "professional.course", "university.degree", "unknown",
    ]
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]
    contact: Literal["cellular", "telephone"]
    month: Literal[
        "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(ge=1, le=100)
    pdays: int = Field(ge=0, le=999)
    previous: int = Field(ge=0, le=20)
    poutcome: Literal["failure", "nonexistent", "success"]

    emp_var_rate: float = Field(alias="emp.var.rate", ge=-10, le=10)
    cons_price_idx: float = Field(alias="cons.price.idx", ge=80, le=110)
    cons_conf_idx: float = Field(alias="cons.conf.idx", ge=-100, le=0)
    euribor3m: float = Field(ge=0, le=10)
    nr_employed: float = Field(alias="nr.employed", ge=4000, le=6000)


class PredictResponse(BaseModel):
    """Structured prediction response returned by the /predict endpoint."""

    model_name: str
    threshold_used: float
    subscribe_probability: float
    subscribe_label: int
    label_meaning: dict[str, str] = {"0": "no", "1": "yes"}