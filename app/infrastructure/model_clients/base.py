"""Abstract interface for model inference clients."""

from __future__ import annotations

from typing import Protocol

from app.domain.entities.inference_result import ModelInferenceResult


class ModelClient(Protocol):
    """Structural interface for a client that calls one inference endpoint.

    Implementations expose their logical ``name`` and ``target_column`` and run
    inference for a job folder, returning a typed :class:`ModelInferenceResult`.
    Implementations must not raise from :meth:`infer`; transport and parsing
    failures are reported via the result's ``error`` field.
    """

    name: str
    target_column: str

    async def infer(self, job_folder: str) -> ModelInferenceResult:
        """Run inference for ``job_folder`` and return a typed result."""
        ...
