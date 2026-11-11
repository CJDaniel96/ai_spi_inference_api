"""Build row-aligned image keys from a single explicit CSV filename template."""

from __future__ import annotations

import math
import string
from typing import Any

import pandas as pd

from app.core.errors import CsvSchemaError

_IMAGE_NAME_COLUMN = "img_name"


def _image_key_part(value: Any) -> str:
    if pd.isna(value):
        raise CsvSchemaError("Image-name template value is missing")
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise CsvSchemaError("Image-name template value is empty")
    return text


class CsvMerger:
    def add_image_name_column(
        self, df: pd.DataFrame, *, template: str, csv_stem: str
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
