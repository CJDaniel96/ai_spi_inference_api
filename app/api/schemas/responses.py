"""Response schemas for the API layer."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health-check response payload.

    Attributes:
        status: Liveness indicator (e.g. ``"healthy"``).
        time: Current UTC+8 timestamp in ISO 8601 format.
    """

    status: str
    time: str


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
    saved_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    csv_count: int = 0
    skipped: Optional[bool] = None
    reason: Optional[str] = None
    img_numbers: Optional[int] = None
