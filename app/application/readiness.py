"""Application: readiness check (config loads + model endpoints probed)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.core.config import AppConfig, get_config
from app.infrastructure.model_clients.health_probe import (
    EndpointHealth,
    probe_endpoints,
)

_Probe = Callable[[AppConfig], Awaitable[list[EndpointHealth]]]


@dataclass
class ReadinessResult:
    """Outcome of a readiness check.

    ``ready`` is gated only on config loading: ``/process`` degrades gracefully
    when a model is down (the failure is reported in the response ``errors``), so
    model reachability is surfaced for diagnostics but does not flip readiness.

    Attributes:
        config_ok: Whether the application config loaded successfully.
        endpoints: Per-endpoint reachability (empty when config failed).
        config_error: Error text when the config failed to load.
    """

    config_ok: bool
    endpoints: list[EndpointHealth] = field(default_factory=list)
    config_error: str | None = None

    @property
    def ready(self) -> bool:
        """Return whether the service is ready to accept jobs."""
        return self.config_ok


async def check_readiness(probe: _Probe | None = None) -> ReadinessResult:
    """Check config validity and probe the enabled model endpoints.

    Args:
        probe: Endpoint prober; defaults to the real HTTP probe. Tests inject a
            fake to avoid network calls.

    Returns:
        A :class:`ReadinessResult`.
    """
    probe_fn = probe or probe_endpoints
    try:
        config = get_config()
    except Exception as exc:  # noqa: BLE001 - report any config failure as not-ready
        return ReadinessResult(config_ok=False, config_error=str(exc))
    return ReadinessResult(config_ok=True, endpoints=await probe_fn(config))
