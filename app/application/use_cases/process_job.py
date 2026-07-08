"""Application use case for processing a job folder.

NOTE: This is a transitional bridge. It currently delegates to the legacy
``ai_server_fastapi.process_folder`` coroutine so the new entry point stays
functional while logic is migrated into the domain and infrastructure layers.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from app.api.schemas.requests import ProcessJobRequest
from app.core.logging import get_logger


class ProcessJobUseCase:
    """Orchestrates end-to-end processing of a single job folder."""

    def __init__(self) -> None:
        """Initialize the use case with a namespaced logger."""
        self._logger = get_logger("process_job")

    async def execute(self, request: ProcessJobRequest) -> Dict[str, Any]:
        """Run the processing pipeline for the given request.

        Args:
            request: The request carrying the job folder path.

        Returns:
            A response dict compatible with
            :class:`app.api.schemas.responses.ProcessJobResponse`.

        TODO(phase-3): Replace the legacy delegation below with calls into the
        domain services (defect classification, CSV merge), infrastructure
        model clients, and output writers.
        """
        # TODO(phase-3): Remove this legacy bridge once logic is migrated.
        # Imported lazily to keep this seam explicit and the module import light.
        from ai_server_fastapi import process_folder  # noqa: PLC0415

        req_id = uuid.uuid4().hex
        self._logger.info(
            "event=process.start req_id=%s job_folder=%s",
            req_id,
            request.job_folder,
        )
        result = await process_folder(request.job_folder, req_id=req_id)
        self._logger.info("event=process.end req_id=%s status=ok", req_id)

        # Preserve the legacy wrapping: pass through an explicit status (e.g. the
        # "finished scanning" skip path), otherwise wrap with status="ok".
        if isinstance(result, dict) and "status" in result:
            return result
        return {"status": "ok", **result}
