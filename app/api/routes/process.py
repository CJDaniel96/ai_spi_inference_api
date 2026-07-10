"""Job-processing route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.api.schemas.requests import ProcessJobRequest
from app.api.schemas.responses import ProcessJobResponse
from app.application.use_cases.process_job import ProcessJobUseCase

router = APIRouter(tags=["process"])


@router.post(
    "/process",
    response_model=ProcessJobResponse,
    response_model_exclude_none=True,
)
async def process(payload: ProcessJobRequest, request: Request) -> dict[str, Any]:
    """Process a job folder and return saved files and errors.

    Delegates to :class:`ProcessJobUseCase`, reusing the lifespan-managed shared
    HTTP client. Typed application errors (missing folder/CSV, CSV schema, output
    write, config) are mapped to HTTP responses by the app-level exception
    handlers registered in :func:`app.main.create_app`; a single model-client
    failure is reported in the ``errors`` array with a 200.

    Args:
        payload: The parsed request body carrying ``job_folder``.
        request: The incoming request (source of the shared HTTP client).

    Returns:
        A dict compatible with :class:`ProcessJobResponse`.
    """
    http_client = getattr(request.app.state, "http_client", None)
    return await ProcessJobUseCase(http_client=http_client).execute(payload)
