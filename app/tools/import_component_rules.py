"""Validate an SX component rules workbook and publish its runtime JSON."""

from __future__ import annotations

import argparse
import re
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from zipfile import BadZipFile

from app.core.component_rules import (
    DEFAULT_COMPONENT_RULE,
    ComponentRule,
    ComponentRules,
)
from app.infrastructure.local_store import atomic_json

RANGE_HEADERS = ("Component", "OBB外觀", "體積", "面積")
LIMIT_HEADERS = ("Component", "OBB外觀", "體積下限", "體積上限", "面積下限", "面積上限")
RANGE_PATTERN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*%\s*[~～]\s*(\d+(?:\.\d+)?)\s*%\s*")


def _text_range(value: object) -> dict[str, float]:
    match = RANGE_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError(f"Expected a range such as 30% ~ 170%, got {value!r}")
    return {"min_pct": float(match[1]), "max_pct": float(match[2])}


def _percent_cell(cell) -> float:
    if type(cell.value) not in (int, float) or "%" not in cell.number_format:
        raise ValueError("All four limits must contain numeric Excel percentages")
    return float(Decimal(str(cell.value)) * 100)


def _read_rules(sheet) -> ComponentRules:
    rows = sheet.iter_rows()
    headers = [cell.value for cell in next(rows, ())]
    while headers and headers[-1] is None:
        headers.pop()
    header = tuple(
        value.strip() if isinstance(value, str) else value for value in headers
    )
    if header not in (RANGE_HEADERS, LIMIT_HEADERS):
        raise ValueError(f"Unexpected headers on {sheet.title!r}: {header!r}")

    rules = {}
    for row_number, row in enumerate(rows, start=2):
        row = row[: len(header)]
        values = [cell.value for cell in row]
        if all(value is None for value in values):
            continue
        try:
            raw_name, appearance, *_ = values
            if (
                not isinstance(raw_name, str)
                or not raw_name.strip()
                or row[0].data_type == "f"
            ):
                raise ValueError("Component must be a nonempty text identifier")
            name = raw_name.strip()
            if name in rules:
                raise ValueError(f"Duplicate component: {name!r}")
            if appearance != "啟用":
                raise ValueError("OBB外觀 must be 啟用")
            if header == RANGE_HEADERS:
                volume, area = map(_text_range, values[2:4])
            else:
                vol_min, vol_max, area_min, area_max = map(_percent_cell, row[2:6])
                volume = {"min_pct": vol_min, "max_pct": vol_max}
                area = {"min_pct": area_min, "max_pct": area_max}
            rules[name] = ComponentRule.model_validate({"volume": volume, "area": area})
        except ValueError as exc:
            raise ValueError(f"{sheet.title} row {row_number}: {exc}") from exc

    if DEFAULT_COMPONENT_RULE not in rules:
        raise ValueError(
            f"{sheet.title} must contain an {DEFAULT_COMPONENT_RULE!r} row"
        )
    return ComponentRules(others=rules.pop(DEFAULT_COMPONENT_RULE), components=rules)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sheet", help="Worksheet name (defaults to the active sheet)")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json":
        parser.error("output must be a .json file")
    try:
        from openpyxl import load_workbook
        from openpyxl.utils.exceptions import InvalidFileException
    except ModuleNotFoundError:
        parser.error("Excel import requires openpyxl (python -m pip install openpyxl)")
    try:
        with closing(load_workbook(args.workbook, read_only=True)) as workbook:
            sheet = workbook[args.sheet] if args.sheet else workbook.active
            if sheet is None:
                raise ValueError("Workbook has no active worksheet")
            rules = _read_rules(sheet)
        atomic_json(args.output, rules.model_dump())
    except (BadZipFile, InvalidFileException, KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(rules.components)} component rules and others to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
