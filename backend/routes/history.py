"""
detecto.routes.history
======================

GET    /history  -- paged, filtered view of stored detections.
DELETE /reset    -- destructive: clears all stored detections.

Filtering happens in SQL (decision D5): the three filters map onto indexed
columns, so we never ship more rows than the client will display.

Timezone note (honest limitation):
    Timestamps are stored in UTC, so the ``time_start`` / ``time_end``
    filters are interpreted as UTC hours, not the operator's local time.
    Adding a caller-supplied timezone is a documented future enhancement.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from backend.models.record import (
    HistoryResponse,
    ResetResponse,
    delete_all,
    query_detections,
)

router = APIRouter(tags=["history"])

# Hard ceiling on a single page, independent of the configured default, so a
# client cannot request an unbounded result set.
HISTORY_MAX_LIMIT = 1000


def _error(status_code: int, code: str, message: str, detail: str | None = None) -> HTTPException:
    """Structured error, mirroring routes/detect.py so the envelope is uniform."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "detail": detail},
    )


def _normalize_datetime(value: Optional[str], *, end_of_day: bool) -> Optional[str]:
    """Parse a date or datetime string into a UTC ISO-8601 string.

    Accepts either ``YYYY-MM-DD`` or a full ISO-8601 datetime. A date-only
    value is expanded to the start (or end) of that day so range filters are
    inclusive of the whole boundary day.
    """
    if value is None:
        return None

    text = value.strip()
    try:
        # Date-only form: expand to the full day.
        if len(text) == 10:
            day = date.fromisoformat(text)
            moment = (
                datetime.combine(day, time.max) if end_of_day
                else datetime.combine(day, time.min)
            )
            return moment.replace(tzinfo=timezone.utc).isoformat()

        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError as exc:
        raise _error(
            400,
            "invalid_datetime",
            f"Could not parse datetime: {value!r}",
            "Use YYYY-MM-DD or a full ISO-8601 datetime",
        ) from exc


def _normalize_time_of_day(value: Optional[str], field_name: str) -> Optional[str]:
    """Validate a HH:MM string. Stored values are exactly this format, so the
    SQL filter is a plain lexicographic comparison."""
    if value is None:
        return None

    text = value.strip()
    try:
        parsed = time.fromisoformat(text)
    except ValueError as exc:
        raise _error(
            400,
            "invalid_time_of_day",
            f"{field_name} must be HH:MM (24-hour)",
            f"Received: {value!r}",
        ) from exc
    return parsed.strftime("%H:%M")


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    request: Request,
    start: Optional[str] = Query(default=None, description="Start date/datetime (inclusive)"),
    end: Optional[str] = Query(default=None, description="End date/datetime (inclusive)"),
    time_start: Optional[str] = Query(default=None, description="Earliest time-of-day HH:MM (UTC)"),
    time_end: Optional[str] = Query(default=None, description="Latest time-of-day HH:MM (UTC)"),
    min_confidence: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Only rows with avg confidence >= this"
    ),
    limit: int = Query(default=0, ge=0, le=HISTORY_MAX_LIMIT, description="0 = use server default"),
    offset: int = Query(default=0, ge=0, description="Rows to skip"),
) -> HistoryResponse:
    """Return stored detections, newest first, matching the given filters."""
    settings = request.app.state.settings

    effective_limit = limit or int(settings.history_default_limit)
    effective_limit = min(effective_limit, HISTORY_MAX_LIMIT)

    start_iso = _normalize_datetime(start, end_of_day=False)
    end_iso = _normalize_datetime(end, end_of_day=True)
    time_start_norm = _normalize_time_of_day(time_start, "time_start")
    time_end_norm = _normalize_time_of_day(time_end, "time_end")

    if start_iso and end_iso and start_iso > end_iso:
        raise _error(400, "invalid_range", "'start' must not be after 'end'")
    if time_start_norm and time_end_norm and time_start_norm > time_end_norm:
        raise _error(400, "invalid_range", "'time_start' must not be after 'time_end'")

    records, total = query_detections(
        settings.db_path,
        start=start_iso,
        end=end_iso,
        time_start=time_start_norm,
        time_end=time_end_norm,
        min_confidence=min_confidence,
        limit=effective_limit,
        offset=offset,
    )

    return HistoryResponse(
        records=records,
        total=total,
        returned=len(records),
        limit=effective_limit,
        offset=offset,
    )


@router.delete("/reset", response_model=ResetResponse)
async def reset_history(request: Request) -> ResetResponse:
    """Delete every stored detection. Destructive and not reversible."""
    settings = request.app.state.settings
    deleted = delete_all(settings.db_path)
    return ResetResponse(
        deleted=deleted,
        message=f"Cleared {deleted} detection record(s). This action cannot be undone.",
    )
