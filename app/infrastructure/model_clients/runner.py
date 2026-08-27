"""Concurrent runner for enabled model clients.

Builds :class:`HttpModelClient` instances from configuration and invokes them in
parallel, isolating individual failures so one bad endpoint cannot abort a job.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence

import httpx

from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.inference_result import ModelInferenceResult
from app.infrastructure.model_clients.base import ModelClient
from app.infrastructure.model_clients.http_model_client import HttpModelClient

# Factory signature for building a client from a config entry (tests inject a
# fake to avoid real HTTP calls).
ClientFactory = Callable[..., ModelClient]


async def _safe_infer(client: ModelClient, job_folder: str) -> ModelInferenceResult:
    """Call ``client.infer`` and turn an unexpected crash into an error result.

    :class:`HttpModelClient` already reports failures via ``error``; this is a
    defensive net so a misbehaving client cannot abort the whole job.
    """
    name = getattr(client, "name", "unknown")
    column = getattr(client, "target_column", "")
    try:
        return await client.infer(job_folder)
    except Exception as exc:  # defensive: isolate one client's crash
        return ModelInferenceResult(
            name=name,
            target_column=column,
            results={},
            request_ms=0.0,
            error=f"Model client '{name}' crashed: {exc}",
        )


async def gather_inferences(
    clients: Sequence[ModelClient],
    job_folder: str,
) -> list[ModelInferenceResult]:
    """Run all ``clients`` concurrently and return their results.

    Args:
        clients: The model clients to invoke.
        job_folder: The job folder path passed to each client.

    Returns:
        One :class:`ModelInferenceResult` per client, in input order. Failed
        calls are represented by results whose ``error`` field is set.
    """
    tasks = [_safe_infer(client, job_folder) for client in clients]
    return list(await asyncio.gather(*tasks))


async def gather_inferences_until_required_complete(
    clients: Sequence[ModelClient],
    required_names: set[str],
    job_folder: str,
    *,
    timeout_seconds: float,
) -> list[ModelInferenceResult]:
    """Return as soon as every required model finishes or the budget expires.

    Optional models are still useful when they finish in parallel, but they are
    cancelled once all required decisions are available so they cannot consume
    the fallback publication reserve.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    started = time.perf_counter()
    tasks = [asyncio.create_task(_safe_infer(client, job_folder)) for client in clients]
    pending = set(tasks)
    completed: dict[str, ModelInferenceResult] = {}
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while pending:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        done, pending = await asyncio.wait(
            pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            break
        for task in done:
            result = task.result()
            completed[result.name] = result
        required_results = [completed.get(name) for name in required_names]
        if required_names and all(result is not None for result in required_results):
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    results: list[ModelInferenceResult] = []
    for _task, client in zip(tasks, clients, strict=True):
        name = getattr(client, "name", "unknown")
        existing = completed.get(name)
        if existing is not None:
            results.append(existing)
            continue
        required = name in required_names
        detail = (
            "exceeded the job inference budget"
            if required
            else "was cancelled after required models completed"
        )
        results.append(
            ModelInferenceResult(
                name=name,
                target_column=getattr(client, "target_column", ""),
                results={},
                request_ms=elapsed_ms,
                error=f"Model client '{name}' {detail}",
            )
        )
    return results


def _build_http_client(
    *,
    name: str,
    url: str,
    target_column: str,
    timeout_seconds: float,
    http_client: httpx.AsyncClient,
    logger: logging.Logger,
    req_id: str | None,
) -> ModelClient:
    """Default factory constructing an :class:`HttpModelClient`."""
    return HttpModelClient(
        name=name,
        url=url,
        target_column=target_column,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
        logger=logger,
        req_id=req_id,
    )


async def _build_and_gather(
    config: AppConfig,
    job_folder: str,
    *,
    factory: ClientFactory,
    logger: logging.Logger,
    req_id: str | None,
    http_client: httpx.AsyncClient,
) -> list[ModelInferenceResult]:
    """Build the enabled clients on ``http_client`` and run them concurrently."""
    clients = [
        factory(
            name=entry.name,
            url=entry.url,
            target_column=entry.target_column,
            timeout_seconds=entry.timeout_seconds,
            http_client=http_client,
            logger=logger,
            req_id=req_id,
        )
        for entry in config.enabled_model_clients()
    ]
    return await gather_inferences(clients, job_folder)


async def _build_and_gather_until_required(
    config: AppConfig,
    job_folder: str,
    *,
    timeout_seconds: float,
    factory: ClientFactory,
    logger: logging.Logger,
    req_id: str | None,
    http_client: httpx.AsyncClient,
) -> list[ModelInferenceResult]:
    clients = [
        factory(
            name=entry.name,
            url=entry.url,
            target_column=entry.target_column,
            timeout_seconds=min(entry.timeout_seconds, timeout_seconds),
            http_client=http_client,
            logger=logger,
            req_id=req_id,
        )
        for entry in config.enabled_model_clients()
    ]
    required_names = {entry.name for entry in config.required_enabled_model_clients()}
    return await gather_inferences_until_required_complete(
        clients,
        required_names,
        job_folder,
        timeout_seconds=timeout_seconds,
    )


async def run_enabled_model_clients(
    config: AppConfig,
    job_folder: str,
    *,
    logger: logging.Logger | None = None,
    req_id: str | None = None,
    client_factory: ClientFactory | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[ModelInferenceResult]:
    """Build and run every enabled model client from ``config`` in parallel.

    Only clients with ``enabled=True`` are invoked. A single client failure does
    not abort the others; its error is carried on its own result.

    Args:
        config: Application config providing the model client entries.
        job_folder: Job folder path forwarded to each client.
        logger: Logger for structured events; defaults to the app logger.
        req_id: Correlation id included in log lines.
        client_factory: Optional factory for building clients (tests inject a
            fake here to avoid real HTTP calls).
        http_client: Optional shared HTTP client (e.g. app lifespan-managed). When
            provided it is reused and left open; otherwise a client is created and
            closed for this call.

    Returns:
        One result per enabled client, in config order.
    """
    log = logger or get_logger("model_client")
    factory = client_factory or _build_http_client
    if http_client is not None:
        return await _build_and_gather(
            config,
            job_folder,
            factory=factory,
            logger=log,
            req_id=req_id,
            http_client=http_client,
        )
    async with httpx.AsyncClient() as owned_client:
        return await _build_and_gather(
            config,
            job_folder,
            factory=factory,
            logger=log,
            req_id=req_id,
            http_client=owned_client,
        )


async def run_enabled_model_clients_until(
    config: AppConfig,
    job_folder: str,
    *,
    timeout_seconds: float,
    logger: logging.Logger | None = None,
    req_id: str | None = None,
    client_factory: ClientFactory | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[ModelInferenceResult]:
    """Run enabled models inside the remaining absolute job budget."""
    log = logger or get_logger("model_client")
    factory = client_factory or _build_http_client
    if http_client is not None:
        return await _build_and_gather_until_required(
            config,
            job_folder,
            timeout_seconds=timeout_seconds,
            factory=factory,
            logger=log,
            req_id=req_id,
            http_client=http_client,
        )
    async with httpx.AsyncClient() as owned_client:
        return await _build_and_gather_until_required(
            config,
            job_folder,
            timeout_seconds=timeout_seconds,
            factory=factory,
            logger=log,
            req_id=req_id,
            http_client=owned_client,
        )
