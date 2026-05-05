"""Staleness guard: abort approved action if a newer drift event has arrived.

Per CLAUDE.md §16 think-about questions:
Q: What if a newer drift event arrives while HIL is pending?
A: Before executing, compare investigation.drift_report_id to the current
   latest report ID. Abort and warn if they differ.
"""

from __future__ import annotations

from core.logging import get_logger
from drift.severity import DriftReport

log = get_logger(__name__)


class StaleInvestigationError(Exception):
    """Raised when a newer drift event supersedes the investigation."""

    def __init__(self, investigation_report_id: str, current_report_id: str) -> None:
        super().__init__(
            f"Investigation opened on report {investigation_report_id!r} "
            f"but current report is {current_report_id!r}. "
            "HIL approval is stale — abort and re-triage."
        )
        self.investigation_report_id = investigation_report_id
        self.current_report_id = current_report_id


async def assert_not_stale(
    investigation_report_id: str,
    current_report: DriftReport,
) -> None:
    """Raise StaleInvestigationError if the investigation is based on an old report.

    Called by the action node in graph.py before dispatching any queue job.

    Args:
        investigation_report_id: report_id stored when the investigation was opened.
        current_report: Latest DriftReport fetched from the model service.

    Raises:
        StaleInvestigationError: If the report IDs differ.
    """
    if investigation_report_id != current_report.report_id:
        log.warning(
            "staleness.detected",
            investigation_report=investigation_report_id,
            current_report=current_report.report_id,
            current_severity=current_report.severity,
        )
        raise StaleInvestigationError(investigation_report_id, current_report.report_id)

    log.info(
        "staleness.ok",
        report_id=investigation_report_id,
        severity=current_report.severity,
    )
