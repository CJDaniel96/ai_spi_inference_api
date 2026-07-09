"""FastAPI server that orchestrates model calls, merges results into CSVs,
and computes derived metrics/labels for SPI jobs.

- Accepts a folder with CSVs/images
- Calls configured model endpoints concurrently
- Merges scalar results back into each CSV and writes to
  `<external_output_root>/<job_folder_name>/AI/*.csv`
- Logs per-request timing/metadata
"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
import uuid

from app.core.config import get_config as get_app_config
from app.core.config import to_legacy_rule_mapping


app = FastAPI(title="AI Merge Server", version="1.0.0")


class JobRequest(BaseModel):
    """Request body carrying the absolute/relative path of a job folder.

    The job folder is expected to contain one or more CSV files (and images)
    that will be enriched and written to an `AI/` subfolder.
    """

    job_folder: str


def get_system_logger() -> logging.Logger:
    """Configure and return the 'system' logger writing to log/system.

    Uses a rotating file handler to avoid unbounded growth (5 MB x 3).
    """
    logger = logging.getLogger("system")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "system"  # no extension as requested

    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log to stderr for visibility when running in foreground
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


# Global exception handler to capture unexpected errors and persist them in system log
@app.exception_handler(Exception)
async def _unhandled_exception_logger(request: Request, exc: Exception):  # noqa: ANN001
    log = get_system_logger()
    # Log full traceback for unexpected errors
    log.exception(
        "event=unhandled.error path=%s method=%s err=%s",
        request.url.path,
        request.method,
        str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


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
    # Guardrails
    folder_images_num_threshold: int


def get_rules() -> RuleConfig:
    """Return the processing thresholds and output roots.

    Adapts the nested application config (`app/core/config.py`) into the legacy
    flat `RuleConfig` shape so existing call sites keep working unchanged.
    Config loading, validation, and caching now live in `app.core.config`.
    """
    return RuleConfig(**to_legacy_rule_mapping(get_app_config()))


async def post_job(
    client: httpx.AsyncClient,
    url: str,
    job_folder: str,
    *,
    timeout: int = 300,
    logger: Optional[logging.Logger] = None,
    req_id: Optional[str] = None,
    service: Optional[str] = None,
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
    log = logger or get_system_logger()
    start = time.perf_counter()
    if service:
        log.info(
            "event=inference.start req_id=%s service=%s url=%s",
            req_id or "-",
            service,
            url,
        )
    try:
        resp = await client.post(url, json={"job_folder": job_folder}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        end = time.perf_counter()
        results = data.get("results", {})
        if not isinstance(results, dict):
            err_msg = f"Unexpected results type from {url}: {type(results)}"
            if service:
                log.error(
                    "event=inference.error req_id=%s service=%s url=%s err=%s request_ms=%.3f",
                    req_id or "-",
                    service,
                    url,
                    err_msg,
                    (end - start) * 1000.0,
                )
            return url, {}, err_msg, {"request_ms": (end - start) * 1000.0}

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

        # Prepare small sample for logging
        try:
            sample_items = list(results.items())[:3]
        except Exception:
            sample_items = []

        if service:
            log.info(
                "event=inference.done req_id=%s service=%s url=%s request_ms=%.3f result_count=%d sample=%s",
                req_id or "-",
                service,
                url,
                (end - start) * 1000.0,
                len(results) if isinstance(results, dict) else -1,
                repr(sample_items),
            )

        return url, results, "", {
            "request_ms": (end - start) * 1000.0,
            "inference_ms": inference_ms,
            "model_version": model_version,
            "device": device,
        }
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        if service:
            log.error(
                "event=inference.error req_id=%s service=%s url=%s err=%s request_ms=%.3f",
                req_id or "-",
                service,
                url,
                str(exc),
                (end - start) * 1000.0,
            )
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


async def process_folder(job_folder: str, *, req_id: Optional[str] = None) -> Dict[str, Any]:
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

    # Model endpoints are driven by config (enabled clients only), not hard-coded.
    enabled_clients = get_app_config().enabled_model_clients()
    endpoints: List[Tuple[str, str]] = [
        (client.url, client.target_column) for client in enabled_clients
    ]
    # Map each endpoint URL to its logical name and per-client request timeout.
    url_to_name: Dict[str, str] = {client.url: client.name for client in enabled_clients}
    url_to_timeout: Dict[str, int] = {
        client.url: client.timeout_seconds for client in enabled_clients
    }

    results_by_endpoint: Dict[str, Dict[str, Any]] = {}
    metrics_by_endpoint: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    # Job-level timing start (around model calls through to file writes)
    request_start_at = _now_tz8_iso()
    job_start = time.perf_counter()

    # Use current year/month for output directory structure
    tz8 = timezone(timedelta(hours=8))
    now = datetime.now(tz8)
    year_str = now.strftime("%Y")
    month_str = now.strftime("%m")

    log = get_system_logger()
    rules = get_rules()

    # Initialize counts for logging
    total_pass = 0
    total_fail = 0
    total_anomaly = 0
    total_distance = 0
    # total_me_thres = 0
    total_low_vol = 0
    total_high_vol = 0
    total_high_cover = 0
    total_high_paste = 0

    img_numbers = _count_images(folder)
    log.info(
        "event=process.pipeline.start req_id=%s job_folder=%s csv_count=%d images=%d",
        req_id or "-",
        str(folder),
        len(csv_files),
        img_numbers,
    )

    # Guard: skip inference if image count exceeds threshold
    try:
        threshold = int(rules.folder_images_num_threshold)
    except Exception:
        threshold = None

    if threshold is not None and img_numbers > threshold:
        job_end = time.perf_counter()
        request_end_at = _now_tz8_iso()

        anomaly_request_ms = 0.0
        paste_request_ms = 0.0
        distance_request_ms = 0.0
        compute_total_ms = 0.0
        request_latency_ms = (job_end - job_start) * 1000.0

        # Also write CSVs to primary and backup with only is_pass=23
        saved_files: List[str] = []
        ai_dir = Path(rules.external_output_root) / folder.name / "AI"
        ai_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = Path(rules.backup_output_root) / year_str / month_str / folder.name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for csv_path in csv_files:
            # 1. Output Phase: Read as string to preserve numeric formatting
            df_final = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            df_final["is_pass"] = "23"
            
            total_fail += len(df_final)
            
            out_path = ai_dir / csv_path.name
            backup_path = backup_dir / csv_path.name
            
            df_final.to_csv(out_path, index=False)
            df_final.to_csv(backup_path, index=False)
            
            saved_files.append(str(out_path))
            saved_files.append(str(backup_path))

            # 2. Processed Phase: Generate *_processed.csv with full AI schema (all NA/Empty)
            df_processing = pd.read_csv(csv_path)
            df_processing["img_name"] = build_img_name_column(df_processing)
            
            # Initialize AI columns from endpoints with NA
            for _, col_name in endpoints:
                df_processing[col_name] = pd.Series([pd.NA] * len(df_processing), dtype="Float64")
            
            df_processing["ai_defect_name"] = ""
            df_processing["is_pass"] = 23
            
            processed_name = f"{csv_path.stem}_processed{csv_path.suffix}"
            processing_backup_path = backup_dir / processed_name
            df_processing.to_csv(processing_backup_path, index=False)
            saved_files.append(str(processing_backup_path))

        log_row = {
            "job_folder": str(folder),
            "img_numbers": int(img_numbers),
            "request_start_at": request_start_at,
            "request_end_at": request_end_at,
            "anomaly_request_ms": anomaly_request_ms,
            "paste_request_ms": paste_request_ms,
            "distance_request_ms": distance_request_ms,
            "compute_total_ms": compute_total_ms,
            "request_latency_ms": request_latency_ms,
            "pass_count": total_pass,
            "fail_count": total_fail,
            "anomaly_count": total_anomaly,
            "distance_count": total_distance,
            # "me_thres_count": total_me_thres,
            "low_vol_count": total_low_vol,
            "high_vol_count": total_high_vol,
            "high_cover_count": total_high_cover,
            "high_paste_count": total_high_paste,
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

        log.info(
            "event=process.skip req_id=%s job_folder=%s reason=images_exceed_threshold images=%d threshold=%s request_latency_ms=%.3f",
            req_id or "-",
            str(folder),
            img_numbers,
            str(threshold),
            request_latency_ms,
        )
        return {
            "status": "finished scanning",
            "skipped": True,
            "reason": "images_exceed_threshold",
            "img_numbers": img_numbers,
            "csv_count": len(csv_files),
            "saved_files": saved_files,
            "errors": [],
        }

    async with httpx.AsyncClient() as client:
        tasks = [
            post_job(
                client,
                url,
                job_folder,
                timeout=url_to_timeout.get(url, 300),
                logger=log,
                req_id=req_id,
                service=url_to_name.get(url, url),
            )
            for url, _ in endpoints
        ]
        for url, res, err, meta in await asyncio.gather(*tasks):
            if err:
                errors.append(err)
            results_by_endpoint[url] = res
            metrics_by_endpoint[url] = meta

    ai_dir = Path(rules.external_output_root) / folder.name / "AI"
    ai_dir.mkdir(parents=True, exist_ok=True)
    # Also create backup directory
    backup_dir = Path(rules.backup_output_root) / year_str / month_str / folder.name 
    backup_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for csv_path in csv_files:
        # 1. Processing Phase: Read normally for calculation
        df = pd.read_csv(csv_path)
        df_processing = df.copy() # Use this for logic
        
        df_processing["img_name"] = build_img_name_column(df_processing)
        for url, col_name in endpoints:
            df_processing = merge_scalar_result(df_processing, results_by_endpoint.get(url, {}), col_name)
        
        # Compute derived metrics and defect name
        # df_processing = add_pad_area_and_cover(df_processing)
        df_processing = add_ai_defect_name(df_processing)
        df_processing = add_is_pass(df_processing)

        # Accumulate counts
        total_pass += int((df_processing["is_pass"] == 22).sum())
        total_fail += int((df_processing["is_pass"] == 23).sum())
        if "ai_defect_name" in df_processing.columns:
            total_anomaly += int((df_processing["ai_defect_name"] == "FM/color").sum())
            total_distance += int((df_processing["ai_defect_name"] == "short distance").sum())
            
            # Individual counts
            total_low_vol += int((df_processing["ai_defect_name"] == "low vol").sum())
            total_high_vol += int((df_processing["ai_defect_name"] == "high vol").sum())
            total_high_cover += int((df_processing["ai_defect_name"] == "high cover").sum())
            total_high_paste += int((df_processing["ai_defect_name"] == "high paste").sum())

            # paste_defects = ["high vol", "low vol", "high cover", "high paste"]
            # total_me_thres += int(df_processing["ai_defect_name"].isin(paste_defects).sum())
        
        print("detect result: ", df_processing.head())
        
        # === MODIFICATION START: Formatting & Saving Phase ===
        # 2. Output Phase: Read again as string to preserve numeric formatting (e.g. "50.000")
        df_final = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        
        # 3. Update ONLY is_pass column using the calculated values
        # Ensure values are cast to string to match the dataframe's dtype
        df_final["is_pass"] = df_processing["is_pass"].astype(str)
        
        out_path = ai_dir / csv_path.name
        backup_path = backup_dir / csv_path.name

        processed_name = f"{csv_path.stem}_processed{csv_path.suffix}"
        processing_backup_path = backup_dir / processed_name        
        
        # Save to primary and backup locations
        df_final.to_csv(out_path, index=False)
        df_final.to_csv(backup_path, index=False)
        df_processing.to_csv(processing_backup_path, index=False)
        # === MODIFICATION END ===
        
        saved_files.append(str(out_path))
        saved_files.append(str(backup_path))
        saved_files.append(str(processing_backup_path))
        
    # Job-level timing end
    job_end = time.perf_counter()
    request_end_at = _now_tz8_iso()

    # Derive per-model metrics mapped by client name (url_to_name is config-derived).
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

    # img_numbers computed earlier

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
        "pass_count": total_pass,
        "fail_count": total_fail,
        "anomaly_count": total_anomaly,
        "distance_count": total_distance,
        # "me_thres_count": total_me_thres,
        "low_vol_count": total_low_vol,
        "high_vol_count": total_high_vol,
        "high_cover_count": total_high_cover,
        "high_paste_count": total_high_paste,
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

    log.info(
        "event=process.summary req_id=%s job_folder=%s csv_count=%d images=%d anomaly_ms=%.3f paste_ms=%.3f distance_ms=%.3f compute_total_ms=%.3f request_latency_ms=%.3f saved_files=%d errors=%d",
        req_id or "-",
        str(folder),
        len(csv_files),
        img_numbers,
        anomaly_request_ms,
        paste_request_ms,
        distance_request_ms,
        compute_total_ms,
        request_latency_ms,
        len(saved_files),
        len(errors),
    )

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
      4) 6.6 > min_pad_distance -> "short distance" (as requested)
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
        mask = (~assigned) & (dist < rules.short_distance_threshold)
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


@app.get("/health")
async def health():
    return {"status": "healthy", "time": _now_tz8_iso()}

@app.post("/process")
async def process_route(req: JobRequest):
    """HTTP endpoint to trigger processing for a given job folder."""
    try:
        log = get_system_logger()
        req_id = uuid.uuid4().hex
        log.info(
            "event=process.start req_id=%s job_folder=%s",
            req_id,
            req.job_folder,
        )
        # Ensure rules are loaded before processing
        _ = get_rules()
        result = await process_folder(req.job_folder, req_id=req_id)
        log.info(
            "event=process.end req_id=%s status=ok",
            req_id,
        )
        # If the processing already provided a status (e.g., finished scanning), pass it through
        if isinstance(result, dict) and "status" in result:
            return result
        return {"status": "ok", **result}
    except FileNotFoundError as e:
        get_system_logger().error(
            "event=process.error req_id=%s status=400 err=%s",
            locals().get("req_id", "-"),
            str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        get_system_logger().error(
            "event=process.error req_id=%s status=400 err=%s",
            locals().get("req_id", "-"),
            str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        # Log unexpected errors with full traceback for post-mortem analysis
        get_system_logger().exception(
            "event=process.error req_id=%s status=500 err=%s",
            locals().get("req_id", "-"),
            str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_server_fastapi:app", host="0.0.0.0", port=5050)

# curl.exe -X POST "http://127.0.0.1:5050/process" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:/Dre/JQ_SPI_02_AI_API/data/20251028142856\" }"
