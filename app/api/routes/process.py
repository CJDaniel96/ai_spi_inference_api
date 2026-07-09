"""Job-processing route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas.requests import ProcessJobRequest
from app.api.schemas.responses import ProcessJobResponse
from app.application.use_cases.process_job import ProcessJobUseCase
from app.core.errors import AppError

router = APIRouter(tags=["process"])


@router.post(
    "/process",
    response_model=ProcessJobResponse,
    response_model_exclude_none=True,
)
async def process(request: ProcessJobRequest) -> dict[str, Any]:
    """Process a job folder and return saved files and errors.

    Delegates to :class:`ProcessJobUseCase`. Error mapping preserves the legacy
    contract: missing folders or CSVs map to HTTP 400, and unexpected failures
    map to HTTP 500.

    Args:
        request: The parsed request body carrying ``job_folder``.

    Returns:
        A dict compatible with :class:`ProcessJobResponse`.

    Raises:
        HTTPException: On any processing error, with a legacy-compatible status.
    """
    use_case = ProcessJobUseCase()
    try:
        return await use_case.execute(request)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as 500 like the legacy route.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
