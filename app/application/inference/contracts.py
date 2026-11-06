"""Preparation contract shared by legacy and native OBB workflows."""

from pathlib import Path
from typing import Protocol

from app.domain.entities.pipeline_result import PipelineResultManifest


class InferencePreparation(Protocol):
    async def prepare(
        self, *, job_id: str, staged_folder: Path, deadline_at: float
    ) -> tuple[PipelineResultManifest, Path]: ...
