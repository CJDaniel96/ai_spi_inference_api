"""Domain entity: model inference results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferenceResult:
    """Scalar results returned by a single model endpoint.

    Attributes:
        service: Logical model name (e.g. ``"anomaly"``, ``"distance"``).
        column: Target CSV column for the scalar value.
        values: Mapping of image name to scalar value (or ``None``).
        error: Error string when the call failed, otherwise empty.
    """

    service: str
    column: str
    values: dict[str, float | None] = field(default_factory=dict)
    error: str = ""

    # TODO(phase-2/3): Produced by HttpModelClient; consumed by CsvMerger.
