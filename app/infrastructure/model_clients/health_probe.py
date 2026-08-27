"""Infrastructure: probe the health of configured model endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.core.config import AppConfig

_HEALTH_PATH = "health"
_DEFAULT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class EndpointHealth:
    """Reachability of a single model endpoint.

    Attributes:
        name: Logical model name.
        url: The probed health URL.
        healthy: True when the endpoint returned 2xx and its payload does not
            report an unready model.
        detail: Short status/error string for diagnostics.
    """

    name: str
    url: str
    healthy: bool
    detail: str


async def probe_endpoints(
    config: AppConfig, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
) -> list[EndpointHealth]:
    """Probe each enabled model client's ``/health`` endpoint.

    Args:
        config: Application config providing the enabled model clients.
        timeout_seconds: Per-probe HTTP timeout.

    Returns:
        One :class:`EndpointHealth` per enabled client. Never raises — an
        unreachable endpoint is reported as ``healthy=False``.
    """
    results: list[EndpointHealth] = []
    async with httpx.AsyncClient() as client:
        for entry in config.enabled_model_clients():
            health_url = urljoin(entry.url, _HEALTH_PATH)
            try:
                response = await client.get(health_url, timeout=timeout_seconds)
                healthy, detail = _response_readiness(response)
            except httpx.HTTPError as exc:
                healthy = False
                detail = str(exc) or exc.__class__.__name__
            results.append(
                EndpointHealth(
                    name=entry.name, url=health_url, healthy=healthy, detail=detail
                )
            )
    return results


def _response_readiness(response: httpx.Response) -> tuple[bool, str]:
    """Interpret both HTTP status and the model server's readiness payload."""
    if not 200 <= response.status_code < 300:
        return False, f"status={response.status_code}"
    try:
        body = response.json()
    except ValueError:
        return True, f"status={response.status_code}"
    if not isinstance(body, dict):
        return True, f"status={response.status_code}"

    status = str(body.get("status", "")).strip().lower()
    model_ready = body.get("model_ready")
    not_ready_statuses = {
        "error",
        "failed",
        "initializing",
        "not_ready",
        "not ready",
        "unhealthy",
    }
    healthy = model_ready is not False and status not in not_ready_statuses
    detail_parts = [f"status={response.status_code}"]
    if status:
        detail_parts.append(f"model_status={status}")
    if model_ready is not None:
        detail_parts.append(f"model_ready={model_ready}")
    return healthy, " ".join(detail_parts)
