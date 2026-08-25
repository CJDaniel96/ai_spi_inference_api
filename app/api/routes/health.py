"""Health and readiness routes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.schemas.responses import HealthResponse, ModelHealth, ReadinessResponse
from app.application.readiness import check_readiness
from app.utils.time_utils import now_tz8_iso

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness and the current UTC+8 timestamp (cheap probe)."""
    return HealthResponse(status="healthy", time=now_tz8_iso())


@router.get("/ready", response_model=ReadinessResponse)
async def ready(response: Response) -> ReadinessResponse:
    """Report readiness: config validity plus model-endpoint reachability.

    Returns 200 when the config loads (the service can accept jobs) and 503 when
    it cannot. Model reachability is reported for diagnostics but does not gate
    readiness, since ``/process`` has a configured all-23 fail-safe policy for a
    required-model failure.
    """
    result = await check_readiness()
    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if result.ready else "not_ready",
        config_ok=result.config_ok,
        config_error=result.config_error,
        models=[
            ModelHealth(
                name=item.name,
                url=item.url,
                healthy=item.healthy,
                detail=item.detail,
            )
            for item in result.endpoints
        ],
    )
