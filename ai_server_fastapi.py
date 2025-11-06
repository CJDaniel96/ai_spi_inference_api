"""FastAPI server that orchestrates model calls, merges results into CSVs,
and computes derived metrics/labels for SPI jobs.

- Accepts a folder with CSVs/images
- Calls configured model endpoints concurrently
- Merges scalar results back into each CSV and writes to
  `<external_output_root>/<job_folder_name>/AI/*.csv`
- Logs per-request timing/metadata
"""

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta


app = FastAPI(title="AI Merge Server", version="1.0.0")


class JobRequest(BaseModel):
    """Request body carrying the absolute/relative path of a job folder.

    The job folder is expected to contain one or more CSV files (and images)
    that will be enriched and written to an `AI/` subfolder.
    """

    job_folder: str


def build_img_name_column(df: pd.DataFrame) -> pd.Series:
    """Build image filename as "{Array_id-1}_{Pad_no}.jpg" for each row.

    Requires columns `Array_id` and `Pad_no`. Raises ValueError if missing.
    """
    if "Array_id" not in df.columns or "Pad_no" not in df.columns:
        missing = [c for c in ("Array_id", "Pad_no") if c not in df.columns]
        raise ValueError(f"CSV missing required columns: {missing}")
    array_minus_one = pd.to_numeric(df["Array_id"], errors="coerce").astype("Int64") - 1
    pad_str = df["Pad_no"].astype(str)
    return array_minus_one.astype(str) + "_" + pad_str + ".jpg"


class RuleConfig(BaseModel):
    """Thresholds, offsets, and output roots used by processing.

    - anomaly_threshold: score above which rows are labeled "FM/color"
    - high_cover_threshold: percent threshold for "high cover"
    - short_distance_threshold: distance threshold for "short distance"
    - low_vol_offset/high_vol_offset: dynamic insp_vol thresholds
    - high_paste_height_threshold: insp_height threshold for "high paste"
    - external_output_root: base folder for enriched CSVs (e.g. "E:/external")
    - backup_output_root: secondary base folder for enriched CSVs
    """

    anomaly_threshold: float
    high_cover_threshold: float
    short_distance_threshold: float
    low_vol_offset: float
    high_vol_offset: float
    high_paste_height_threshold: float
    external_output_root: str
    backup_output_root: str


_RULES: RuleConfig | None = None


def _load_rules_from_file(path: Path) -> RuleConfig:
    """Load `RuleConfig` from a JSON file at `path`."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RuleConfig(**data)


def get_rules() -> RuleConfig:
    """Return memoized rules, reading once from env or default config.

    Uses env var `AI_RULES_PATH` when set; otherwise reads `config/ai_server.json`.
    """
    global _RULES
    if _RULES is not None:
        return _RULES
    # Path can be overridden via env var AI_RULES_PATH
    env_path = os.getenv("AI_RULES_PATH")
    cfg_path = Path(env_path) if env_path else Path("config/ai_server.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Rules config not found: {cfg_path}")
    _RULES = _load_rules_from_file(cfg_path)
    return _RULES


async def post_job(
    client: httpx.AsyncClient, url: str, job_folder: str, timeout: int = 300
) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
    """Call a model endpoint and capture timings/metadata.

    Returns: (url, results_map, error_str, meta)
      - results_map: dict of img_name -> float
      - meta: {
          'request_ms': float,
          'inference_ms': Optional[float],   # parsed from response if available
          'model_version': Optional[str],
          'device': Optional[str],
        }
    """
    start = time.perf_counter()
    try:
        resp = await client.post(url, json={"job_folder": job_folder}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        end = time.perf_counter()
        results = data.get("results", {})
        if not isinstance(results, dict):
            return url, {}, f"Unexpected results type from {url}: {type(results)}", {
                "request_ms": (end - start) * 1000.0
            }

        # Try to extract optional inference/metadata from common keys if provided by the model server
        inference_ms = None
        model_version = None
        device = None
        if isinstance(data, dict):
            inference_ms = (
                data.get("inference_ms")
                or (isinstance(data.get("metrics"), dict) and data.get("metrics", {}).get("inference_ms"))
                or (isinstance(data.get("timings"), dict) and data.get("timings", {}).get("inference_ms"))
            )
            model_version = (
                data.get("model_version")
                or data.get("model_ver")
                or data.get("version")
            )
            device = data.get("device")

        return url, results, "", {
            "request_ms": (end - start) * 1000.0,
            "inference_ms": inference_ms,
            "model_version": model_version,
            "device": device,
        }
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return url, {}, f"Request to {url} failed: {exc}", {"request_ms": (end - start) * 1000.0}

def _now_tz8_iso() -> str:
    """Return current time in UTC+8 as an ISO 8601 string."""
    tz8 = timezone(timedelta(hours=8))
    return datetime.now(tz8).isoformat()

def _count_images(job_folder: Path) -> int:
    """Recursively count images under `job_folder` with common extensions."""
    # Count image files recursively with common extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return sum(1 for p in job_folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def merge_scalar_result(df: pd.DataFrame, results: Dict[str, Any], column_name: str) -> pd.DataFrame:
    """Merge scalar float results into df under the specified column name.

    - results is a dict keyed by img_name -> float (or dict containing column_name)
    - coerces to numeric floats, non-parsable values become NaN
    """
    if not results:
        if column_name not in df.columns:
            df[column_name] = pd.Series([pd.NA] * len(df), dtype="Float64")
        return df

    values: Dict[str, Any] = {}
    for k, v in results.items():
        if isinstance(v, dict):
            if column_name in v:
                values[k] = v.get(column_name)
        else:
            values[k] = v

    if not values:
        if column_name not in df.columns:
            df[column_name] = pd.Series([pd.NA] * len(df), dtype="Float64")
        return df

    map_df = pd.DataFrame({"img_name": list(values.keys()), column_name: list(values.values())})
    map_df[column_name] = pd.to_numeric(map_df[column_name], errors="coerce")
    merged = df.merge(map_df, on="img_name", how="left")
    # Ensure dtype is numeric float for the merged column
    merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce")
    return merged


async def process_folder(job_folder: str) -> Dict[str, Any]:
    """End-to-end pipeline for a job folder.

    - Validates folder and finds CSVs
    - Calls all model endpoints concurrently and collects results/metrics
    - Merges scalar results into each CSV and computes derived columns
    - Writes enriched CSVs to `<external_output_root>/<job_folder_name>/AI/`
      and appends a request log row
    Returns a dict with saved file paths, error messages, and CSV count.
    """
    folder = Path(job_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {job_folder}")

    csv_files: List[Path] = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {job_folder}")

    # Map each model endpoint to its target float column name
    endpoints: List[Tuple[str, str]] = [
        ("http://localhost:8000/inference", "anomaly_score"),
        ("http://127.0.0.1:8001/inference", "paste_pixels"),
        ("http://127.0.0.1:8002/inference", "min_pad_distance"),
    ]

    results_by_endpoint: Dict[str, Dict[str, Any]] = {}
    metrics_by_endpoint: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    # Job-level timing start (around model calls through to file writes)
    request_start_at = _now_tz8_iso()
    job_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [post_job(client, url, job_folder) for url, _ in endpoints]
        for url, res, err, meta in await asyncio.gather(*tasks):
            if err:
                errors.append(err)
            results_by_endpoint[url] = res
            metrics_by_endpoint[url] = meta

    # Output enriched CSVs under configured external root: <root>/<job_name>/AI
    rules = get_rules()
    ai_dir = Path(rules.external_output_root) / folder.name / "AI"
    ai_dir.mkdir(parents=True, exist_ok=True)
    # Also create backup directory
    backup_dir = Path(rules.backup_output_root) / folder.name / "AI"
    backup_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df = df.copy()
        df["img_name"] = build_img_name_column(df)
        for url, col_name in endpoints:
            df = merge_scalar_result(df, results_by_endpoint.get(url, {}), col_name)
        # Compute derived metrics and defect name
        df = add_pad_area_and_cover(df)
        df = add_ai_defect_name(df)
        df = add_is_pass(df)
        out_path = ai_dir / csv_path.name
        backup_path = backup_dir / csv_path.name

        # Save to primary and backup locations
        df.to_csv(out_path, index=False)
        df.to_csv(backup_path, index=False)
        saved_files.append(str(out_path))
        saved_files.append(str(backup_path))

    # Job-level timing end
    job_end = time.perf_counter()
    request_end_at = _now_tz8_iso()

    # Derive per-model metrics mapped by target column
    url_to_name = {endpoints[0][0]: "anomaly", endpoints[1][0]: "paste", endpoints[2][0]: "distance"}

    def get_metric(prefix: str, key: str) -> float:
        for url, _col in endpoints:
            if url_to_name.get(url) == prefix:
                meta = metrics_by_endpoint.get(url, {})
                val = meta.get(key)
                if val is None:
                    return float("nan")
                try:
                    return float(val)
                except Exception:
                    return float("nan")
        return float("nan")

    anomaly_request_ms = get_metric("anomaly", "request_ms")
    paste_request_ms = get_metric("paste", "request_ms")
    distance_request_ms = get_metric("distance", "request_ms")
    # Removed per-endpoint inference_ms metrics from logging per request

    compute_total_ms = sum(
        v for v in [anomaly_request_ms, paste_request_ms, distance_request_ms] if not pd.isna(v)
    )
    request_latency_ms = (job_end - job_start) * 1000.0

    # Count images present in folder (jpg/jpeg)
    img_numbers = _count_images(folder)

    # Append or create job-level log CSV in repository-level log folder
    log_row = {
        "job_folder": str(folder),
        "img_numbers": int(img_numbers),
        "request_start_at": request_start_at,
        "request_end_at": request_end_at,
        "anomaly_request_ms": anomaly_request_ms,
        "paste_request_ms": paste_request_ms,
        "distance_request_ms": distance_request_ms,
        # inference_ms fields removed from log row
        "compute_total_ms": compute_total_ms,
        "request_latency_ms": request_latency_ms,
        "logged_at": _now_tz8_iso(),
    }
    log_df = pd.DataFrame([log_row])
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.csv"
    if log_path.exists():
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    return {"saved_files": saved_files, "errors": errors, "csv_count": len(csv_files)}


# ------------------------
# Post-merge enrich helpers
# ------------------------

def add_pad_area_and_cover(df: pd.DataFrame) -> pd.DataFrame:
    """Add pad_area and cover% columns when possible.

    pad_area = pi * Width * Length / 4
    cover% = paste_pixels * 0.8246 * 100 / pad_area
    """
    out = df.copy()
    # pad_area
    if "Width" in out.columns and "Length" in out.columns:
        width = pd.to_numeric(out["Width"], errors="coerce")
        length = pd.to_numeric(out["Length"], errors="coerce")
        out["pad_area"] = math.pi * width * length / 4.0
    # cover%
    if "paste_pixels" in out.columns and "pad_area" in out.columns:
        paste_pixels = pd.to_numeric(out["paste_pixels"], errors="coerce")
        out["cover%"] = paste_pixels * 0.8246 * 100.0 / out["pad_area"]
    return out


def add_ai_defect_name(df: pd.DataFrame) -> pd.DataFrame:
    """Add ai_defect_name column per row using ordered rules.

    Priority:
      1) anomaly_score > 0.5 -> "FM/color"
      2) insp_vol outside dynamic thresholds -> "high vol" / "low vol"
      3) cover% > 180 -> "high cover"
      4) 6.6 < min_pad_distance -> "short distance" (as requested)
      5) insp_height > 200 -> "high paste"
    Only the first matching condition per row is applied.
    """
    out = df.copy()
    if "ai_defect_name" not in out.columns:
        out["ai_defect_name"] = ""

    assigned = out["ai_defect_name"].astype(str).str.len() > 0

    rules = get_rules()

    # 1) Anomaly threshold
    if "anomaly_score" in out.columns:
        anomaly = pd.to_numeric(out["anomaly_score"], errors="coerce")
        mask = (~assigned) & (anomaly > rules.anomaly_threshold)
        out.loc[mask, "ai_defect_name"] = "FM/color"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 2) insp_vol thresholds using offsets and vol_l_ng, vol_h_ng
    cols_needed = {"insp_vol", "vol_l_ng", "vol_h_ng"}
    if cols_needed.issubset(out.columns):
        insp_vol = pd.to_numeric(out["insp_vol"], errors="coerce")
        vol_l_ng = pd.to_numeric(out["vol_l_ng"], errors="coerce")
        vol_h_ng = pd.to_numeric(out["vol_h_ng"], errors="coerce")
        low_thr = vol_l_ng + rules.low_vol_offset
        high_thr = vol_h_ng + rules.high_vol_offset

        mask_high = (~assigned) & (insp_vol > high_thr)
        out.loc[mask_high, "ai_defect_name"] = "high vol"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

        mask_low = (~assigned) & (insp_vol < low_thr)
        out.loc[mask_low, "ai_defect_name"] = "low vol"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 3) high cover
    if "cover%" in out.columns:
        cover = pd.to_numeric(out["cover%"], errors="coerce")
        mask = (~assigned) & (cover > rules.high_cover_threshold)
        out.loc[mask, "ai_defect_name"] = "high cover"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 4) short distance (note: condition provided as 6.6 < min_pad_distance)
    if "min_pad_distance" in out.columns:
        dist = pd.to_numeric(out["min_pad_distance"], errors="coerce")
        mask = (~assigned) & (dist > rules.short_distance_threshold)
        out.loc[mask, "ai_defect_name"] = "short distance"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 5) high paste based on insp_height
    if "insp_height" in out.columns:
        height = pd.to_numeric(out["insp_height"], errors="coerce")
        mask = (~assigned) & (height > rules.high_paste_height_threshold)
        out.loc[mask, "ai_defect_name"] = "high paste"

    return out


def add_is_pass(df: pd.DataFrame) -> pd.DataFrame:
    """Add/update 'is_pass' column based on ai_defect_name.

    - If ai_defect_name == "" (empty after strip) => is_pass = 22
    - Else => is_pass = 23
    """
    out = df.copy()
    if "ai_defect_name" not in out.columns:
        out["ai_defect_name"] = ""
    # default 22
    is_pass = pd.Series(22, index=out.index, dtype="Int64")
    mask_fail = out["ai_defect_name"].astype(str).str.strip() != ""
    is_pass.loc[mask_fail] = 23
    out["is_pass"] = is_pass
    return out


@app.post("/process")
async def process_route(req: JobRequest):
    """HTTP endpoint to trigger processing for a given job folder."""
    try:
        # Ensure rules are loaded before processing
        _ = get_rules()
        result = await process_folder(req.job_folder)
        return {"status": "ok", **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_server_fastapi:app", host="0.0.0.0", port=5050, reload=False)

# curl.exe -X POST "http://127.0.0.1:5050/process" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:/Dre/JQ_SPI_02_AI_API/data/20251028142856\" }"
