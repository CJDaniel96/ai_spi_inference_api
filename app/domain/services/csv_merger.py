"""Domain service: image-name building and model-result merging.

Pure pandas logic — no filesystem, HTTP, or FastAPI dependencies.
"""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from typing import Any

import pandas as pd

from app.core.errors import CsvSchemaError
from app.domain.entities.inference_result import ModelInferenceResult

_IMAGE_SUFFIX = ".jpg"
_IMAGE_NAME_COLUMN = "img_name"
_ARRAY_ID_COLUMN = "Array_id"
_PAD_NO_COLUMN = "Pad_no"


def _image_key_part(value: Any) -> str:
    """Return a filename-safe textual representation of one CSV value."""
    if pd.isna(value):
        raise CsvSchemaError("Image-name template value is missing")
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise CsvSchemaError("Image-name template value is empty")
    return text


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

    def add_image_name_column(
        self,
        df: pd.DataFrame,
        *,
        source_column: str | None = None,
        template: str | None = None,
        csv_stem: str | None = None,
    ) -> pd.DataFrame:
        """Return a copy of ``df`` with the ``img_name`` column added.

        ``template`` supports a machine-specific filename built from
        ``{csv_stem}`` and CSV column names. With no template,
        ``source_column`` may name a column that already carries the image path.
        With neither configured, the legacy ``{Array_id - 1}_{Pad_no}.jpg``
        convention is retained.

        Args:
            df: Input frame.
            source_column: Optional column containing image filenames/paths.
            template: Optional image filename format string.
            csv_stem: Source CSV filename without its extension.

        Returns:
            A new frame with an ``img_name`` column.

        Raises:
            CsvSchemaError: If the selected input columns are missing or empty.
        """
        if template is not None:
            return self._add_from_template(df, template=template, csv_stem=csv_stem)

        if source_column is not None:
            if source_column not in df.columns:
                raise CsvSchemaError(
                    f"CSV missing configured image-name column: {source_column!r}"
                )
            source = df[source_column]
            missing = source.isna() | source.astype(str).str.strip().eq("")
            if missing.any():
                row_numbers = [
                    position + 2
                    for position, is_missing in enumerate(missing.tolist())
                    if is_missing
                ][:5]
                raise CsvSchemaError(
                    f"CSV image-name column {source_column!r} is empty at "
                    f"CSV row(s): {row_numbers}"
                )
            out = df.copy()
            out[_IMAGE_NAME_COLUMN] = source.astype(str).map(
                lambda value: value.replace("\\", "/").rsplit("/", 1)[-1]
            )
            return out

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

    @staticmethod
    def _add_from_template(
        df: pd.DataFrame, *, template: str, csv_stem: str | None
    ) -> pd.DataFrame:
        """Build ``img_name`` from ``csv_stem`` and configured CSV columns."""
        formatter = string.Formatter()
        fields = {
            field_name
            for _, field_name, _, _ in formatter.parse(template)
            if field_name
        }
        allowed_fields = set(df.columns) | {"csv_stem"}
        missing_fields = sorted(fields - allowed_fields)
        if missing_fields:
            raise CsvSchemaError(
                f"Image-name template references missing fields: {missing_fields}"
            )
        if "csv_stem" in fields and not csv_stem:
            raise CsvSchemaError("Image-name template requires the CSV filename")

        image_names: list[str] = []
        for position, (_, row) in enumerate(df.iterrows(), start=2):
            values = {
                column: _image_key_part(row[column])
                for column in fields
                if column != "csv_stem"
            }
            values["csv_stem"] = csv_stem or ""
            try:
                image_names.append(template.format_map(values))
            except (KeyError, ValueError) as exc:
                raise CsvSchemaError(
                    f"Unable to build image filename at CSV row {position}: {exc}"
                ) from exc

        out = df.copy()
        out[_IMAGE_NAME_COLUMN] = image_names
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
