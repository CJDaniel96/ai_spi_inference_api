"""Domain service: derived metric columns (``pad_area``, ``cover%``).

Pure pandas logic. Missing input columns yield warnings, never exceptions, so a
disabled paste model (no ``paste_pixels``) cannot break the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# pad_area = pi * Width * Length / _PAD_AREA_DIVISOR
_PAD_AREA_DIVISOR = 4.0
# cover% = paste_pixels * _PASTE_AREA_RATIO * _PERCENT / pad_area
_PASTE_AREA_RATIO = 0.8246
_PERCENT = 100.0

_WIDTH_COLUMN = "Width"
_LENGTH_COLUMN = "Length"
_PASTE_PIXELS_COLUMN = "paste_pixels"
_PAD_AREA_COLUMN = "pad_area"
_COVER_COLUMN = "cover%"


@dataclass
class DerivedMetricsResult:
    """Outcome of derived-column computation.

    Attributes:
        df: The frame with any computable derived columns added.
        warnings: Human-readable notes for each skipped column.
    """

    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


class DerivedMetricsCalculator:
    """Computes ``pad_area`` and ``cover%`` when their input columns exist."""

    def add_derived_columns(self, df: pd.DataFrame) -> DerivedMetricsResult:
        """Add ``pad_area`` and/or ``cover%`` when their inputs are present.

        Never raises on missing columns; each skipped column is recorded as a
        warning so the caller can log it.

        Args:
            df: Input frame.

        Returns:
            A :class:`DerivedMetricsResult` with the enriched frame and warnings.
        """
        out = df.copy()
        warnings: list[str] = []

        if _WIDTH_COLUMN in out.columns and _LENGTH_COLUMN in out.columns:
            width = pd.to_numeric(out[_WIDTH_COLUMN], errors="coerce")
            length = pd.to_numeric(out[_LENGTH_COLUMN], errors="coerce")
            out[_PAD_AREA_COLUMN] = math.pi * width * length / _PAD_AREA_DIVISOR
        else:
            warnings.append(
                f"'{_PAD_AREA_COLUMN}' skipped: requires "
                f"'{_WIDTH_COLUMN}' and '{_LENGTH_COLUMN}'"
            )

        if _PASTE_PIXELS_COLUMN in out.columns and _PAD_AREA_COLUMN in out.columns:
            paste_pixels = pd.to_numeric(out[_PASTE_PIXELS_COLUMN], errors="coerce")
            out[_COVER_COLUMN] = (
                paste_pixels * _PASTE_AREA_RATIO * _PERCENT / out[_PAD_AREA_COLUMN]
            )
        else:
            warnings.append(
                f"'{_COVER_COLUMN}' skipped: requires "
                f"'{_PASTE_PIXELS_COLUMN}' and '{_PAD_AREA_COLUMN}'"
            )

        return DerivedMetricsResult(df=out, warnings=warnings)
