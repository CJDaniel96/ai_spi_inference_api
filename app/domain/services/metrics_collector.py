"""Domain service: per-job pass/fail and defect-count aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.domain.entities.defect import FAIL_CODE, PASS_CODE, DefectLabel

_DEFECT_COLUMN = "ai_defect_name"
_IS_PASS_COLUMN = "is_pass"


@dataclass
class JobCounts:
    """Counters accumulated across a job's CSV files.

    Attributes:
        pass_count: Rows with ``is_pass == PASS_CODE`` (22).
        fail_count: Rows with ``is_pass == FAIL_CODE`` (23).
        defect_counts: Per-label counts keyed by ``ai_defect_name`` value.
    """

    pass_count: int = 0
    fail_count: int = 0
    defect_counts: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Accumulates pass/fail and defect counts from classified frames."""

    def __init__(self) -> None:
        """Initialize an empty accumulator with all defect labels at zero."""
        self._counts = JobCounts(
            defect_counts={label.value: 0 for label in DefectLabel}
        )

    def accumulate(self, df: pd.DataFrame) -> None:
        """Fold the pass/fail and defect counts of ``df`` into the totals.

        Args:
            df: A classified frame with ``is_pass`` and ``ai_defect_name``.
        """
        if _IS_PASS_COLUMN in df.columns:
            self._counts.pass_count += int((df[_IS_PASS_COLUMN] == PASS_CODE).sum())
            self._counts.fail_count += int((df[_IS_PASS_COLUMN] == FAIL_CODE).sum())
        if _DEFECT_COLUMN in df.columns:
            labels = df[_DEFECT_COLUMN].astype(str)
            for label in DefectLabel:
                self._counts.defect_counts[label.value] += int(
                    (labels == label.value).sum()
                )

    @property
    def counts(self) -> JobCounts:
        """Return the accumulated counts."""
        return self._counts
