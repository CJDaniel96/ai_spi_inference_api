"""Unit tests for the concurrent model client runner (fake clients only)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.core.config import load_config
from app.domain.entities.inference_result import ModelInferenceResult
from app.infrastructure.model_clients.runner import (
    gather_inferences,
    run_enabled_model_clients,
)

_REAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ai_server.json"


class _FakeClient:
    """Minimal ModelClient stand-in for exercising the runner."""

    def __init__(
        self,
        name: str,
        target_column: str,
        *,
        result: ModelInferenceResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.target_column = target_column
        self._result = result
        self._raises = raises

    async def infer(self, job_folder: str) -> ModelInferenceResult:
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


def _ok(name: str, column: str) -> ModelInferenceResult:
    """Build a successful result carrying a single scalar value."""
    return ModelInferenceResult(
        name=name,
        target_column=column,
        results={"a.jpg": 1.0},
        request_ms=1.0,
    )


def test_single_failure_does_not_block_other_results() -> None:
    anomaly = _FakeClient(
        "anomaly", "anomaly_score", result=_ok("anomaly", "anomaly_score")
    )
    failing = _FakeClient("distance", "min_pad_distance", raises=RuntimeError("boom"))
    paste = _FakeClient("paste", "paste_pixels", result=_ok("paste", "paste_pixels"))

    results = asyncio.run(gather_inferences([anomaly, failing, paste], "/job"))
    by_name = {result.name: result for result in results}

    assert by_name["anomaly"].results == {"a.jpg": 1.0}
    assert by_name["anomaly"].error is None
    # The crashing client is isolated: it yields an error result, not an exception.
    assert by_name["distance"].error is not None
    assert by_name["distance"].results == {}
    # A successful client still returns its results for merging.
    assert by_name["paste"].results == {"a.jpg": 1.0}
    assert by_name["paste"].error is None
    assert len(results) == 3


def test_gather_preserves_order_and_target_columns() -> None:
    anomaly = _FakeClient(
        "anomaly", "anomaly_score", result=_ok("anomaly", "anomaly_score")
    )
    distance = _FakeClient(
        "distance", "min_pad_distance", result=_ok("distance", "min_pad_distance")
    )

    results = asyncio.run(gather_inferences([anomaly, distance], "/job"))

    assert [r.name for r in results] == ["anomaly", "distance"]
    assert [r.target_column for r in results] == ["anomaly_score", "min_pad_distance"]


def test_run_enabled_model_clients_builds_only_enabled() -> None:
    config = load_config(_REAL_CONFIG_PATH)
    built: list[str] = []

    def fake_factory(*, name: str, target_column: str, **_: object) -> _FakeClient:
        built.append(name)
        return _FakeClient(name, target_column, result=_ok(name, target_column))

    results = asyncio.run(
        run_enabled_model_clients(config, "/job", client_factory=fake_factory)
    )

    # paste is disabled in the shipped config, so it is never built or called.
    assert built == ["anomaly", "distance"]
    assert [r.name for r in results] == ["anomaly", "distance"]


def test_run_enabled_model_clients_reuses_injected_http_client() -> None:
    config = load_config(_REAL_CONFIG_PATH)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"results": {}})

    async def _run() -> list[ModelInferenceResult]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            results = await run_enabled_model_clients(
                config, "/job", http_client=client
            )
            # An injected client is reused and NOT closed by the runner.
            assert client.is_closed is False
            return results

    results = asyncio.run(_run())

    assert [r.name for r in results] == ["anomaly", "distance"]
    assert len(calls) == 2
