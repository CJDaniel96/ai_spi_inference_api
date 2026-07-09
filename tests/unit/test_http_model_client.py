"""Unit tests for the HTTP model client (no real endpoints; mock transport)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.domain.entities.inference_result import ModelInferenceResult
from app.infrastructure.model_clients.http_model_client import HttpModelClient

_Handler = Callable[[httpx.Request], httpx.Response]


async def _infer(
    handler: _Handler,
    *,
    name: str = "anomaly",
    target_column: str = "anomaly_score",
    timeout_seconds: int = 5,
) -> ModelInferenceResult:
    """Run one ``infer`` against a mock transport driven by ``handler``."""
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HttpModelClient(
            name=name,
            url="http://model/inference",
            target_column=target_column,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
        )
        return await client.infer("/job/folder")


def _run(coro: Awaitable[Any]) -> Any:
    """Execute an awaitable to completion without pytest-asyncio."""
    return asyncio.run(coro)


def test_parses_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {"0_1.jpg": 0.5, "0_2.jpg": None}})

    result = _run(_infer(handler))

    assert result.error is None
    assert result.results == {"0_1.jpg": 0.5, "0_2.jpg": None}
    assert result.request_ms >= 0.0


def test_missing_results_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    result = _run(_infer(handler))

    assert result.error is not None
    assert "results" in result.error
    assert result.results == {}


def test_non_dict_results_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [1, 2, 3]})

    result = _run(_infer(handler))

    assert result.error is not None
    assert result.results == {}


def test_timeout_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    result = _run(_infer(handler))

    assert result.error is not None
    assert "failed" in result.error.lower()
    assert result.results == {}


def test_http_status_error_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    result = _run(_infer(handler))

    assert result.error is not None
    assert result.results == {}


def test_target_column_preserved_on_success_and_error() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {"a.jpg": 1.0}})

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    ok_result = _run(_infer(ok, target_column="anomaly_score"))
    err_result = _run(_infer(bad, target_column="min_pad_distance"))

    assert ok_result.target_column == "anomaly_score"
    assert err_result.target_column == "min_pad_distance"


def test_parses_optional_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": {"a.jpg": 1.0},
                "inference_ms": 12.5,
                "version": "v3",
                "device": "cuda:0",
            },
        )

    result = _run(_infer(handler))

    assert result.inference_ms == 12.5
    assert result.model_version == "v3"
    assert result.device == "cuda:0"
