"""Tests for semantic model readiness payload handling."""

from __future__ import annotations

import httpx
import pytest

from app.infrastructure.model_clients.health_probe import _response_readiness


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "error"},
        {"status": "initializing", "model_ready": False},
        {"status": "healthy", "model_ready": False},
    ],
)
def test_http_200_is_not_ready_when_model_payload_says_not_ready(payload) -> None:
    response = httpx.Response(200, json=payload)

    healthy, detail = _response_readiness(response)

    assert healthy is False
    assert "status=200" in detail


def test_healthy_model_payload_is_ready() -> None:
    response = httpx.Response(
        200,
        json={"status": "healthy", "model_ready": True},
    )

    healthy, detail = _response_readiness(response)

    assert healthy is True
    assert "model_status=healthy" in detail


def test_non_success_http_status_is_not_ready() -> None:
    response = httpx.Response(503, json={"status": "initializing"})

    healthy, detail = _response_readiness(response)

    assert healthy is False
    assert detail == "status=503"
