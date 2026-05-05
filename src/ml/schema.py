from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BankMarketingRequest(BaseModel):
    """
    Validated single prediction request for the Bank Marketing model.

    This schema represents the columns the model expects at inference time,
    after removing the leakage column: duration.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    age: int = Field(ge=17, le=100)
    #age must be a number between 17 and 100

    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ] #job must be one of the known job categories

    marital: Literal[
        "divorced",
        "married",
        "single",
        "unknown",
    ]

    education: Literal[
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ]

    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]

    contact: Literal["cellular", "telephone"]

    month: Literal[
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]  #month must be one of the months in the dataset

    day_of_week: Literal[
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
    ]

    campaign: int = Field(ge=1, le=100)
    pdays: int = Field(ge=0, le=999)
    previous: int = Field(ge=0, le=20)

    poutcome: Literal[
        "failure",
        "nonexistent",
        "success",
    ]

    emp_var_rate: float = Field(alias="emp.var.rate", ge=-10, le=10)
    cons_price_idx: float = Field(alias="cons.price.idx", ge=80, le=110)
    cons_conf_idx: float = Field(alias="cons.conf.idx", ge=-100, le=0)
    euribor3m: float = Field(ge=0, le=10)
    nr_employed: float = Field(alias="nr.employed", ge=4000, le=6000)


def to_api_error(exc) -> dict:
    """
    Convert Pydantic validation errors into clean API-style errors.
    """
    return {
        "error": "validation_error",
        "details": [
            {
                "loc": list(error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ],
    }