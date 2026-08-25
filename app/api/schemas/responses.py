"""Response schemas for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health-check response payload.

    Attributes:
        status: Liveness indicator (e.g. ``"healthy"``).
        time: Current UTC+8 timestamp in ISO 8601 format.
    """

    status: str
    time: str


class ModelHealth(BaseModel):
    """Reachability of one model endpoint (readiness diagnostics)."""

    name: str
    url: str
    healthy: bool
    detail: str


class ReadinessResponse(BaseModel):
    """Readiness-check response payload.

    ``status`` is ``"ready"`` when the config loaded (HTTP 200) or ``"not_ready"``
    when it did not (HTTP 503). ``models`` reports endpoint reachability for
    diagnostics only and does not affect readiness.
    """

    status: str
    config_ok: bool
    config_error: str | None = None
    models: list[ModelHealth] = Field(default_factory=list)


class ProcessJobResponse(BaseModel):
    """Response payload for the ``/process`` endpoint.

    Mirrors the legacy contract: successful runs return ``status="ok"`` with
    the saved-file list, while skipped runs (e.g. image count over threshold)
    additionally carry ``skipped``/``reason``/``img_numbers``. The route uses
    ``response_model_exclude_none=True`` so ``None`` fields are dropped, keeping
    the serialized shape byte-compatible with ``ai_server_fastapi.py``.
    """

    model_config = ConfigDict(extra="allow")

    status: str
    saved_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    csv_count: int = 0
    skipped: bool | None = None
    fallback: bool | None = None
    reason: str | None = None
    img_numbers: int | None = None
    primary_return_latency_ms: float | None = None
    deadline_met: bool | None = None
