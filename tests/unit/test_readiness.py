"""Unit tests for the readiness check."""

from __future__ import annotations

import asyncio

import pytest

from app.application import readiness as readiness_module
from app.application.readiness import check_readiness
from app.infrastructure.model_clients.health_probe import EndpointHealth


async def _probe_healthy(config: object) -> list[EndpointHealth]:
    return [
        EndpointHealth(name="anomaly", url="http://x/health", healthy=True, detail="ok")
    ]


async def _probe_down(config: object) -> list[EndpointHealth]:
    return [
        EndpointHealth(name="anomaly", url="http://x/health", healthy=False, detail="x")
    ]


def test_ready_when_config_ok() -> None:
    result = asyncio.run(check_readiness(probe=_probe_healthy))
    assert result.ready is True
    assert result.config_ok is True
    assert result.endpoints[0].healthy is True


def test_ready_even_when_model_down() -> None:
    result = asyncio.run(check_readiness(probe=_probe_down))
    assert result.ready is True
    assert result.endpoints[0].healthy is False


def test_not_ready_when_config_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise RuntimeError("bad config")

    monkeypatch.setattr(readiness_module, "get_config", _raise)

    result = asyncio.run(check_readiness(probe=_probe_healthy))

    assert result.ready is False
    assert result.config_ok is False
    assert "bad config" in (result.config_error or "")
