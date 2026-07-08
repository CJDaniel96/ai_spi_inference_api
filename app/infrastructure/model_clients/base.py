"""Abstract base for model inference clients."""

from __future__ import annotations

import abc

from app.domain.entities.inference_result import InferenceResult


class ModelClient(abc.ABC):
    """Interface for calling an external inference endpoint."""

    @abc.abstractmethod
    async def infer(self, job_folder: str) -> InferenceResult:
        """Run inference for a job folder and return scalar results."""
        raise NotImplementedError
