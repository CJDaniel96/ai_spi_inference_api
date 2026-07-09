"""Domain service: per-job metrics aggregation (skeleton)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JobMetrics:
    """Counters accumulated across a job's CSV files.

    Attributes:
        pass_count: Rows with ``is_pass == 22``.
        fail_count: Rows with ``is_pass == 23``.
        defect_counts: Per-label counts keyed by ``ai_defect_name``.
    """

    pass_count: int = 0
    fail_count: int = 0
    defect_counts: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Accumulates metrics and prepares the ``log.csv`` row.

    TODO(phase-3): Migrate the timing/count aggregation and log-row assembly
    from ``ai_server_fastapi.process_folder`` here.
    """

    def __init__(self) -> None:
        """Initialize an empty metrics accumulator."""
        self._metrics = JobMetrics()
