"""A failed cold restart leaves durable jobs available to Publisher."""

import asyncio
from types import SimpleNamespace

import pytest

from app.application.inference.worker import InferenceWorker


def test_cold_recovery_precedes_claim_and_failure_keeps_no_lease():
    events = []

    def claim(**kwargs):
        events.append("claim")
        return None

    async def warm():
        events.append("warm")
        await asyncio.sleep(0)

    worker = InferenceWorker(
        config=SimpleNamespace(pipeline=SimpleNamespace(worker_lease_seconds=60)),
        repository=SimpleNamespace(claim_for_inference=claim),
        service=None,
        worker_id="native",
        prepare_runtime=warm,
    )
    assert asyncio.run(worker.run_once()) is False
    assert events == ["warm", "claim"]
    events.clear()

    async def failed_warmup():
        events.append("warm")
        raise RuntimeError("GPU model could not restart")

    worker._prepare_runtime = failed_warmup
    with pytest.raises(RuntimeError, match="restart"):
        asyncio.run(worker.run_once())
    assert events == ["warm"]
