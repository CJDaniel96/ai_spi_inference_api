"""Domain service: image-name building and model-result merging.

Pure pandas logic — no filesystem, HTTP, or FastAPI dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from app.core.errors import CsvSchemaError
from app.domain.entities.inference_result import ModelInferenceResult

_IMAGE_SUFFIX = ".jpg"
_IMAGE_NAME_COLUMN = "img_name"
_ARRAY_ID_COLUMN = "Array_id"
_PAD_NO_COLUMN = "Pad_no"


def build_image_name(array_id: Any, pad_no: Any) -> str:
    """Build the image filename ``"{Array_id - 1}_{Pad_no}.jpg"`` for one row.

    Args:
        array_id: The 1-based array id; converted to a number and decremented
            to form the 0-based filename prefix.
        pad_no: The pad number; kept verbatim as a string.

    Returns:
        The image filename, e.g. ``"0_1426.jpg"`` for ``array_id=1, pad_no=1426``.

    Raises:
        ValueError: If ``array_id`` is missing/not numeric, or ``pad_no`` is
            missing.
    """
    numeric_id = pd.to_numeric(array_id, errors="coerce")
    if pd.isna(numeric_id):
        raise CsvSchemaError(f"Array_id is missing or not numeric: {array_id!r}")
    if pd.isna(pad_no):
        raise CsvSchemaError(f"Pad_no is missing: {pad_no!r}")
    prefix = int(numeric_id) - 1
    return f"{prefix}_{pad_no}{_IMAGE_SUFFIX}"


class CsvMerger:
    """Adds the image-name column and merges model results into a DataFrame."""

    def add_image_name_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with the ``img_name`` column added.

        Mirrors the legacy vectorized build ``{Array_id - 1}_{Pad_no}.jpg``.

        Args:
            df: Input frame; must contain ``Array_id`` and ``Pad_no`` columns.

        Returns:
            A new frame with an ``img_name`` column.

        Raises:
            CsvSchemaError: If ``Array_id`` or ``Pad_no`` columns are missing.
        """
        missing = [
            column
            for column in (_ARRAY_ID_COLUMN, _PAD_NO_COLUMN)
            if column not in df.columns
        ]
        if missing:
            raise CsvSchemaError(f"CSV missing required columns: {missing}")

        out = df.copy()
        prefix = (
            pd.to_numeric(out[_ARRAY_ID_COLUMN], errors="coerce").astype("Int64") - 1
        )
        pad_str = out[_PAD_NO_COLUMN].astype(str)
        out[_IMAGE_NAME_COLUMN] = prefix.astype(str) + "_" + pad_str + _IMAGE_SUFFIX
        return out

    def merge_model_results(
        self,
        df: pd.DataFrame,
        model_results: Sequence[ModelInferenceResult],
    ) -> pd.DataFrame:
        """Merge each model result's scalar values into ``df`` by image name.

        The target column for each result is taken from the result itself, never
        hard-coded. A missing/empty result yields an all-``NaN`` column rather
        than an error, and image keys absent from ``df`` are ignored.

        Args:
            df: Frame containing an ``img_name`` column.
            model_results: Results to merge; each provides its ``target_column``.

        Returns:
            A new frame with one column per model result.

        Raises:
            CsvSchemaError: If ``df`` has no ``img_name`` column.
        """
        if _IMAGE_NAME_COLUMN not in df.columns:
            raise CsvSchemaError(
                f"DataFrame has no '{_IMAGE_NAME_COLUMN}' column; "
                "call add_image_name_column first"
            )
        out = df
        for result in model_results:
            out = self._merge_one(out, result.results, result.target_column)
        return out

    @staticmethod
    def _merge_one(
        df: pd.DataFrame,
        results: dict[str, float | None],
        column_name: str,
    ) -> pd.DataFrame:
        """Merge one ``{img_name: value}`` mapping into ``column_name``."""
        if not results:
            out = df.copy()
            if column_name not in out.columns:
                out[column_name] = pd.Series([pd.NA] * len(out), dtype="Float64")
            return out

        map_df = pd.DataFrame(
            {
                _IMAGE_NAME_COLUMN: list(results.keys()),
                column_name: list(results.values()),
            }
        )
        map_df[column_name] = pd.to_numeric(map_df[column_name], errors="coerce")
        merged = df.merge(map_df, on=_IMAGE_NAME_COLUMN, how="left")
        merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce")
        return merged
