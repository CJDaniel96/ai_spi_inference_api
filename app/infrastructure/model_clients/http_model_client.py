"""HTTP implementation of the :class:`ModelClient` interface.

Migrated from the legacy ``ai_server_fastapi.post_job``: issues the POST, times
the request, parses ``results``, extracts optional metadata, and reports any
transport/parse failure via the returned result's ``error`` field rather than
raising.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.logging import get_logger
from app.domain.entities.inference_result import ModelInferenceResult

# Maximum number of result items included in a "done" log sample.
_MAX_SAMPLE_ITEMS = 3
# Fallback timeout (seconds) used when a non-positive value is configured.
_DEFAULT_TIMEOUT_SECONDS = 300


def _elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since ``start`` (a ``perf_counter`` value)."""
    return (time.perf_counter() - start) * 1000.0


def _extract_metadata(
    data: dict[str, Any],
) -> tuple[float | None, str | None, str | None]:
    """Extract optional ``(inference_ms, model_version, device)`` from a body."""
    inference_ms = data.get("inference_ms")
    if inference_ms is None:
        for section in ("metrics", "timings"):
            block = data.get(section)
            if isinstance(block, dict) and block.get("inference_ms") is not None:
                inference_ms = block["inference_ms"]
                break
    model_version = (
        data.get("model_version") or data.get("model_ver") or data.get("version")
    )
    device = data.get("device")
    return inference_ms, model_version, device


class HttpModelClient:
    """Calls a single model ``POST /inference`` endpoint over HTTP.

    The URL, target column, and timeout come from configuration. The shared
    :class:`httpx.AsyncClient` is injected so the caller owns its lifecycle and
    tests can supply a mock transport. :meth:`infer` never raises; failures are
    captured on the returned result's ``error`` field.
    """

    def __init__(
        self,
        *,
        name: str,
        url: str,
        target_column: str,
        timeout_seconds: int,
        http_client: httpx.AsyncClient,
        logger: logging.Logger | None = None,
        req_id: str | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            name: Logical model name (also the metrics/log prefix).
            url: The model's ``POST /inference`` endpoint.
            target_column: CSV column the returned scalar is merged into.
            timeout_seconds: Per-request HTTP timeout.
            http_client: Shared async HTTP client used to issue the request.
            logger: Logger for structured events; defaults to the app logger.
            req_id: Correlation id included in log lines.
        """
        self.name = name
        self.target_column = target_column
        self._url = url
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds > 0 else _DEFAULT_TIMEOUT_SECONDS
        )
        self._http_client = http_client
        self._logger = logger or get_logger("model_client")
        self._req_id = req_id

    async def infer(self, job_folder: str) -> ModelInferenceResult:
        """Call the endpoint for ``job_folder`` and return a typed result."""
        self._log_start()
        start = time.perf_counter()
        try:
            response = await self._http_client.post(
                self._url,
                json={"job_folder": job_folder},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            return self._failure(f"Request to {self._url} failed: {exc}", start)
        except ValueError as exc:  # non-JSON / malformed response body
            return self._failure(f"Invalid JSON from {self._url}: {exc}", start)

        return self._parse(data, _elapsed_ms(start))

    def _parse(self, data: Any, request_ms: float) -> ModelInferenceResult:
        """Validate the response body and build a success or error result."""
        if not isinstance(data, dict):
            return self._error_result(
                f"Unexpected response type from {self._url}: {type(data).__name__}",
                request_ms,
            )
        if "results" not in data:
            return self._error_result(
                f"Response from {self._url} missing 'results'", request_ms
            )
        results = data["results"]
        if not isinstance(results, dict):
            return self._error_result(
                f"Unexpected results type from {self._url}: {type(results).__name__}",
                request_ms,
            )

        inference_ms, model_version, device = _extract_metadata(data)
        result = ModelInferenceResult(
            name=self.name,
            target_column=self.target_column,
            results=results,
            request_ms=request_ms,
            inference_ms=inference_ms,
            model_version=model_version,
            device=device,
            error=None,
        )
        self._log_done(result)
        return result

    def _failure(self, message: str, start: float) -> ModelInferenceResult:
        """Build an error result for a transport failure (timing from ``start``)."""
        return self._error_result(message, _elapsed_ms(start))

    def _error_result(self, message: str, request_ms: float) -> ModelInferenceResult:
        """Log and build an error result with empty scalar values."""
        self._log_error(message, request_ms)
        return ModelInferenceResult(
            name=self.name,
            target_column=self.target_column,
            results={},
            request_ms=request_ms,
            error=message,
        )

    def _log_start(self) -> None:
        """Emit the ``inference.start`` event."""
        self._logger.info(
            "event=inference.start req_id=%s service=%s url=%s",
            self._req_id or "-",
            self.name,
            self._url,
        )

    def _log_done(self, result: ModelInferenceResult) -> None:
        """Emit the ``inference.done`` event with a small result sample."""
        sample = list(result.results.items())[:_MAX_SAMPLE_ITEMS]
        self._logger.info(
            "event=inference.done req_id=%s service=%s url=%s "
            "request_ms=%.3f result_count=%d sample=%s",
            self._req_id or "-",
            self.name,
            self._url,
            result.request_ms,
            len(result.results),
            repr(sample),
        )

    def _log_error(self, message: str, request_ms: float) -> None:
        """Emit the ``inference.error`` event."""
        self._logger.error(
            "event=inference.error req_id=%s service=%s url=%s err=%s request_ms=%.3f",
            self._req_id or "-",
            self.name,
            self._url,
            message,
            request_ms,
        )
