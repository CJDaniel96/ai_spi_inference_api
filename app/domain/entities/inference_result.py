"""Domain entity / DTO for a single model endpoint's inference result."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInferenceResult:
    """Typed, immutable result of one model endpoint call.

    Attributes:
        name: Logical model name (e.g. ``"anomaly"``, ``"distance"``).
        target_column: CSV column the scalar values are merged into.
        results: Mapping of image name to scalar value (or ``None``).
        request_ms: Wall-clock request latency in milliseconds.
        inference_ms: Server-reported inference time, when provided.
        model_version: Server-reported model version, when provided.
        device: Server-reported device, when provided.
        error: Error description when the call failed, otherwise ``None``.
    """

    name: str
    target_column: str
    results: dict[str, float | None] = field(default_factory=dict)
    request_ms: float = 0.0
    inference_ms: float | None = None
    model_version: str | None = None
    device: str | None = None
    error: str | None = None
