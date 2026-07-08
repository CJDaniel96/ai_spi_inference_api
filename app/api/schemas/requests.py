"""Request schemas for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessJobRequest(BaseModel):
    """Request body for triggering job-folder processing.

    Attributes:
        job_folder: Absolute or relative path to the job folder containing the
            CSV files and images to process.
    """

    job_folder: str = Field(..., description="Path to the job folder to process.")
