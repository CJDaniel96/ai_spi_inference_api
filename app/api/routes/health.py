"""Health-check route."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.responses import HealthResponse
from app.utils.time_utils import now_tz8_iso

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness and the current UTC+8 timestamp."""
    return HealthResponse(status="healthy", time=now_tz8_iso())
