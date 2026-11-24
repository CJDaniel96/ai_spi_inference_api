"""Export five-unit histograms of CSV limit excesses for appearance-AI OK images.

Standard library only: this file can also run outside the project checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath

TZ8 = timezone(timedelta(hours=8))
METRICS = {
    "volume": ("體積", "%", "insp_vol", "vol_l_ng", "vol_h_ng"),
    "area": ("面積", "%", "insp_area", "area_l_ng", "area_h_ng"),
    "height": ("高度", "%", "insp_hei", "hei_l_ng", "hei_h_ng"),
    "shift": ("位移", "mm", "insp_x", "insp_y", "shift_x_ng", "shift_y_ng"),
}
HISTOGRAM_FIELDS = [
    "metric",
    "metric_zh",
    "unit",
    "bin_index",
    "bin_range",
    "excess_gt",
    "excess_le",
    "low_ng_count",
    "high_ng_count",
    "ng_count",
]
DETAIL_FIELDS = [
    "manifest_path",
    "date",
    "job_id",
    "source_csv",
    "processed_csv",
    "csv_row",
    "img_name",
    "result_code",
    "failed_rule",
    "metric",
    "metric_zh",
    "unit",
    "direction",
    "value",
    "low",
    "high",
    "excess",
    "bin_range",
    "x_mm",
    "y_mm",
    "x_limit_mm",
    "y_limit_mm",
]
ERROR_FIELDS = ["manifest_path", "source_csv", "csv_row", "error"]


@dataclass
class Statistics:
    bins: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    errors: list[dict] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


def number(value, label):
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid number: {label}={value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"Non-finite number: {label}={value!r}")
    return parsed


def positive_number(value):
    try:
        parsed = number(value, "bin width")
        if parsed <= 0:
            raise ValueError("Bin width must be greater than zero")
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def decimal_text(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def bucket(excess, width):
    """Use (0, width], (width, 2*width], ... without rounding measurements."""
    index = int((excess / width).to_integral_value(rounding=ROUND_CEILING)) - 1
    if not 0 <= index < 10000:
        raise ValueError("Excess requires over 10000 bins; increase the bin width")
    low, high = index * width, (index + 1) * width
    return index, f"({decimal_text(low)},{decimal_text(high)}]"


def safe_child(root, relative):
    if not isinstance(relative, str):
        raise ValueError("CSV path must be a string")
    relative = relative.replace("\\", "/")
    candidate = PurePosixPath(relative)
    if (
        not candidate.parts
        or candidate.is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in candidate.parts
    ):
        raise ValueError(f"Unsafe CSV path: {relative!r}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"CSV path leaves manifest directory: {relative!r}")
    return path


def load_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    if data.get("outcome") not in {"normal", "fallback"}:
        raise ValueError("Unknown manifest outcome")
    if not isinstance(data.get("job_id"), str) or not data["job_id"].strip():
        raise ValueError("Missing manifest job_id")
    timestamp = float(number(data["created_at"], "created_at"))
    day = datetime.fromtimestamp(timestamp, TZ8).date()
    items = data["csv_results"]
    if not isinstance(items, list) or not items:
        raise ValueError("csv_results must be a nonempty list")
    sources, processed = set(), set()
    for item in items:
        source = safe_child(path.parent, item["source_csv"])
        target = safe_child(path.parent, item["processed_csv"])
        if source in sources or target in processed:
            raise ValueError("Duplicate source_csv or processed_csv")
        sources.add(source)
        processed.add(target)
        codes = item["result_codes"]
        if not isinstance(codes, list) or any(
            type(code) is not int or code not in (22, 23) for code in codes
        ):
            raise ValueError("result_codes must contain only integers 22/23")
    return data, day


def read_processed(manifest_path, item):
    root = manifest_path.parent
    source = safe_child(root, item["processed_csv"])
    if not source.is_file():
        # MiNiFi moves the original processed CSV here after exporting it.
        source = safe_child(root, f"exported/{source.name}")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"is_pass", "ai_rule_checks", "ai_failed_rule"}
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV columns")
        if not required.issubset(fields):
            raise ValueError(f"Missing audit columns: {sorted(required - set(fields))}")
        rows = list(reader)
    if len(rows) != len(item["result_codes"]):
        raise ValueError("Processed CSV row count differs from manifest result_codes")
    # Reject a stale/misaligned CSV as a whole before including any of its rows.
    for line, (row, code) in enumerate(zip(rows, item["result_codes"], strict=True), 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {line}")
        if number(row["is_pass"], f"row {line} is_pass") != code:
            raise ValueError(f"CSV is_pass disagrees with manifest at row {line}")
    return source, rows


def appearance_ok(row):
    checks = json.loads(row["ai_rule_checks"])
    if not isinstance(checks, list) or any(not isinstance(c, dict) for c in checks):
        raise ValueError("ai_rule_checks must be a list of objects")
    appearance = [c for c in checks if c.get("rule") == "yolo_obb"]
    if len(appearance) != 1 or type(appearance[0].get("passed")) is not bool:
        raise ValueError("Missing or ambiguous yolo_obb passed flag")
    return appearance[0]["passed"]


def metric_excess(row, metric):
    definition = METRICS[metric]
    values = [number(row.get(column), column) for column in definition[2:]]
    if metric == "shift":
        x, y, x_limit, y_limit = values
        if x_limit < 0 or y_limit < 0:
            raise ValueError("Shift limits must be nonnegative")
        excess = max(abs(x) - x_limit, abs(y) - y_limit)
        return (
            {
                "direction": "high",
                "excess": excess,
                "x_mm": x,
                "y_mm": y,
                "x_limit_mm": x_limit,
                "y_limit_mm": y_limit,
            }
            if excess > 0
            else None
        )
    value, left, right = values
    low, high = min(left, right), max(left, right)
    if low <= value <= high:
        return None
    return {
        "direction": "low" if value < low else "high",
        "value": value,
        "low": low,
        "high": high,
        "excess": low - value if value < low else value - high,
    }


def row_excesses(row, code, mode):
    failed_rule = row["ai_failed_rule"].strip()
    if not appearance_ok(row):
        if code != 23 or failed_rule != "yolo_obb":
            raise ValueError("Appearance NG disagrees with final result/failed rule")
        return None
    if code == 22 and failed_rule or code == 23 and failed_rule not in METRICS:
        raise ValueError("Appearance OK disagrees with final failed rule")
    selected = [failed_rule] if mode == "first-failure" and code == 23 else METRICS
    excesses = {}
    for metric in selected:
        result = metric_excess(row, metric)
        if result:
            excesses[metric] = result
    first = next(iter(excesses), "")
    if (code == 23) != bool(excesses) or first != failed_rule:
        raise ValueError("Measurements disagree with final result/failed rule")
    return excesses


def collect(
    backup,
    start=None,
    end=None,
    *,
    width=Decimal(5),
    shift_width=None,
    mode="all",
    keep_details=False,
):
    result = Statistics()
    shift_width = width if shift_width is None else shift_width
    paths = [backup] if backup.is_file() else sorted(backup.rglob("manifest.json"))
    for path in paths:
        context = {"manifest_path": str(path), "source_csv": "", "csv_row": ""}
        try:
            manifest, day = load_manifest(path)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            result.errors.append({**context, "error": str(exc)})
            continue
        if start and day < start or end and day > end:
            continue
        result.counts["matched_manifests"] += 1
        if manifest["outcome"] == "fallback":
            result.counts["fallback_manifests"] += 1
            continue
        result.counts["normal_manifests"] += 1
        for item in manifest["csv_results"]:
            context["source_csv"] = item["source_csv"]
            context["csv_row"] = ""
            try:
                source, rows = read_processed(path, item)
            except (OSError, ValueError, csv.Error) as exc:
                result.counts["invalid_csv_files"] += 1
                result.errors.append({**context, "error": str(exc)})
                continue
            result.counts["valid_csv_files"] += 1
            for line, (row, code) in enumerate(
                zip(rows, item["result_codes"], strict=True), 2
            ):
                result.counts["total_rows"] += 1
                context["csv_row"] = line
                try:
                    excesses = row_excesses(row, code, mode)
                    events = []
                    for metric, values in (excesses or {}).items():
                        step = shift_width if metric == "shift" else width
                        index, label = bucket(values["excess"], step)
                        events.append((metric, values, index, label))
                except (ValueError, TypeError, OverflowError) as exc:
                    result.counts["invalid_rows"] += 1
                    result.errors.append({**context, "error": str(exc)})
                    continue
                if excesses is None:
                    result.counts["appearance_ng_rows"] += 1
                    continue
                result.counts["appearance_ok_rows"] += 1
                result.counts["numeric_ng_images"] += bool(excesses)
                for metric, values, index, label in events:
                    result.bins[metric, index, values["direction"]] += 1
                    result.counts["numeric_ng_events"] += 1
                    if keep_details:
                        result.details.append(
                            {
                                **{
                                    key: decimal_text(value)
                                    if isinstance(value, Decimal)
                                    else value
                                    for key, value in values.items()
                                },
                                **{key: context[key] for key in context},
                                "date": day.isoformat(),
                                "job_id": manifest["job_id"],
                                "processed_csv": str(source),
                                "img_name": row.get("img_name", ""),
                                "result_code": code,
                                "failed_rule": row["ai_failed_rule"],
                                "metric": metric,
                                "metric_zh": METRICS[metric][0],
                                "unit": METRICS[metric][1],
                                "bin_range": label,
                            }
                        )
    return result


def histogram_rows(statistics, width=Decimal(5), shift_width=None):
    shift_width = width if shift_width is None else shift_width
    for metric, (name, unit, *_) in METRICS.items():
        step = shift_width if metric == "shift" else width
        last = max((i for m, i, _ in statistics.bins if m == metric), default=0)
        for index in range(last + 1):
            low, high = index * step, (index + 1) * step
            below = statistics.bins[metric, index, "low"]
            above = statistics.bins[metric, index, "high"]
            yield {
                "metric": metric,
                "metric_zh": name,
                "unit": unit,
                "bin_index": index + 1,
                "bin_range": f"({decimal_text(low)},{decimal_text(high)}]",
                "excess_gt": decimal_text(low),
                "excess_le": decimal_text(high),
                "low_ng_count": below,
                "high_ng_count": above,
                "ng_count": below + above,
            }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path, help="Backup directory or manifest.json")
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Inclusive Taiwan date, YYYY-MM-DD (created_at)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Inclusive Taiwan date, YYYY-MM-DD (created_at)",
    )
    parser.add_argument("--bin-width", type=positive_number, default=Decimal(5))
    parser.add_argument(
        "--shift-bin-width",
        type=positive_number,
        help="Shift bin width in mm; defaults to --bin-width",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "first-failure"),
        default="all",
        help="All out-of-range metrics or recorded first failure",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("measurement_statistics.csv")
    )
    parser.add_argument("--details", type=Path, help="Optional per-image excess CSV")
    args = parser.parse_args(argv)
    root = args.backup.expanduser().resolve()
    if not root.exists():
        parser.error(f"Backup does not exist: {root}")
    if root.is_file() and root.name != "manifest.json":
        parser.error("Input file must be named manifest.json")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("start-date must be <= end-date")
    output = args.output.expanduser().resolve()
    errors_path = output.with_name(f"{output.stem}_errors.csv")
    details = args.details.expanduser().resolve() if args.details else None
    targets = [output, errors_path, *([details] if details else [])]
    if len(set(targets)) != len(targets):
        parser.error("Output, details and errors CSV paths must be different")
    backup_dir = root.parent if root.is_file() else root
    for target in targets:
        if target.suffix.lower() != ".csv":
            parser.error(f"Output must have a .csv extension: {target}")
        if target.is_relative_to(backup_dir):
            parser.error("Reports must be outside the backup input directory")
        if target.exists() and not target.is_file():
            parser.error(f"Output is not a file: {target}")
    result = collect(
        root,
        args.start_date,
        args.end_date,
        width=args.bin_width,
        shift_width=args.shift_bin_width,
        mode=args.mode,
        keep_details=details is not None,
    )
    write_csv(
        output,
        histogram_rows(result, args.bin_width, args.shift_bin_width),
        HISTOGRAM_FIELDS,
    )
    if details:
        write_csv(details, result.details, DETAIL_FIELDS)
    write_csv(errors_path, result.errors, ERROR_FIELDS)
    for name in (
        "matched_manifests",
        "normal_manifests",
        "fallback_manifests",
        "valid_csv_files",
        "invalid_csv_files",
        "total_rows",
        "appearance_ok_rows",
        "appearance_ng_rows",
        "invalid_rows",
        "numeric_ng_images",
        "numeric_ng_events",
    ):
        print(f"{name}: {result.counts[name]}")
    print(f"Histogram: {output}\nErrors: {errors_path}")
    if not result.counts["appearance_ok_rows"]:
        print("No valid appearance-AI OK rows matched; check coverage before charting.")
    return 1 if result.errors or not result.counts["appearance_ok_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
