"""Classifier and ref-stats dependencies: pull singletons from app.state."""

from __future__ import annotations

from fastapi import Request
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ml.reference_stats import ReferenceStats


def get_classifier(request: Request) -> Pipeline:
    """Return the production classifier loaded at startup."""
    return request.app.state.classifier  # type: ignore[return-value]


def get_threshold(request: Request) -> float:
    """Return the operating decision threshold loaded at startup."""
    return float(request.app.state.threshold)


def get_ref_stats(request: Request) -> ReferenceStats:
    """Return the reference statistics loaded at startup."""
    return request.app.state.ref_stats  # type: ignore[return-value]
