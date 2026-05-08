"""Classifier and ref-stats dependencies: pull singletons from app.state."""

from __future__ import annotations

from fastapi import HTTPException, Request
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ml.reference_stats import ReferenceStats


def get_classifier(request: Request) -> Pipeline:
    """Return the production classifier loaded at startup."""
    if request.app.state.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run `make train` first.",
        )
    return request.app.state.classifier  # type: ignore[return-value]


def get_threshold(request: Request) -> float:
    """Return the operating decision threshold loaded at startup."""
    return float(request.app.state.threshold)


def get_ref_stats(request: Request) -> ReferenceStats:
    """Return the reference statistics loaded at startup."""
    if request.app.state.ref_stats is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run `make train` first.",
        )
    return request.app.state.ref_stats  # type: ignore[return-value]
