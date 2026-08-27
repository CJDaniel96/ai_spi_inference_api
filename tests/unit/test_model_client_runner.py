"""Unit tests for the concurrent model client runner (fake clients only)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.core.config import load_config
from app.domain.entities.inference_result import ModelInferenceResult
from app.infrastructure.model_clients.runner import (
    gather_inferences,
    gather_inferences_until_required_complete,
    run_enabled_model_clients,
    run_enabled_model_clients_until,
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


class _SlowClient(_FakeClient):
    def __init__(self, name: str, target_column: str, *, delay: float) -> None:
        super().__init__(name, target_column, result=_ok(name, target_column))
        self._delay = delay
        self.cancelled = False

    async def infer(self, job_folder: str) -> ModelInferenceResult:
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return await super().infer(job_folder)


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


def test_optional_model_is_cancelled_after_required_models_complete() -> None:
    required = _FakeClient(
        "required", "required_score", result=_ok("required", "required_score")
    )
    optional = _SlowClient("optional", "optional_score", delay=1)

    results = asyncio.run(
        gather_inferences_until_required_complete(
            [required, optional], {"required"}, "/job", timeout_seconds=0.5
        )
    )

    by_name = {result.name: result for result in results}
    assert by_name["required"].error is None
    assert "cancelled after required models completed" in by_name["optional"].error
    assert optional.cancelled is True


def test_required_model_timeout_is_returned_as_error_result() -> None:
    required = _SlowClient("required", "required_score", delay=1)
    optional = _FakeClient(
        "optional", "optional_score", result=_ok("optional", "optional_score")
    )

    results = asyncio.run(
        gather_inferences_until_required_complete(
            [required, optional], {"required"}, "/job", timeout_seconds=0.02
        )
    )

    by_name = {result.name: result for result in results}
    assert "exceeded the job inference budget" in by_name["required"].error
    assert by_name["optional"].error is None
    assert required.cancelled is True


def test_deadline_runner_caps_each_http_timeout_to_remaining_budget() -> None:
    config = load_config(_REAL_CONFIG_PATH)
    configured_timeouts: list[float] = []

    def fake_factory(
        *, name: str, target_column: str, timeout_seconds: float, **_: object
    ) -> _FakeClient:
        configured_timeouts.append(timeout_seconds)
        return _FakeClient(name, target_column, result=_ok(name, target_column))

    asyncio.run(
        run_enabled_model_clients_until(
            config,
            "/job",
            timeout_seconds=0.5,
            client_factory=fake_factory,
        )
    )

    assert configured_timeouts == [0.5, 0.5]
