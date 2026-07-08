"""HTTP implementation of :class:`ModelClient` (skeleton)."""

from __future__ import annotations

from app.domain.entities.inference_result import InferenceResult
from app.infrastructure.model_clients.base import ModelClient


class HttpModelClient(ModelClient):
    """Calls a model ``/inference`` endpoint over HTTP.

    TODO(phase-2/3): Migrate ``ai_server_fastapi.post_job`` here (httpx call,
    timing capture, error handling). Endpoint URLs must come from config rather
    than the hard-coded values currently embedded in the legacy server.
    """

    def __init__(self, service: str, url: str, column: str) -> None:
        """Initialize with the logical service name, endpoint URL, and column.

        Args:
            service: Logical model name (e.g. ``"anomaly"``).
            url: Full ``/inference`` endpoint URL.
            column: Target CSV column for the returned scalar.
        """
        self._service = service
        self._url = url
        self._column = column

    async def infer(self, job_folder: str) -> InferenceResult:
        """Call the endpoint for ``job_folder`` and return its scalar results."""
        raise NotImplementedError("Migrated in a later phase.")
