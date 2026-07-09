"""FastAPI server that orchestrates model calls, merges results into CSVs,
and computes derived metrics/labels for SPI jobs.

- Accepts a folder with CSVs/images
- Calls configured model endpoints concurrently
- Merges scalar results back into each CSV and writes to
  `<external_output_root>/<job_folder_name>/AI/*.csv`
- Logs per-request timing/metadata
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
import uuid

from app.core.config import get_config as get_app_config
from app.core.config import to_legacy_rule_mapping
from app.domain.services.csv_merger import CsvMerger
from app.domain.services.defect_classifier import DefectClassifier, add_is_pass
from app.domain.services.derived_metrics import DerivedMetricsCalculator
from app.infrastructure.model_clients.runner import run_enabled_model_clients


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


def _now_tz8_iso() -> str:
    """Return current time in UTC+8 as an ISO 8601 string."""
    tz8 = timezone(timedelta(hours=8))
    return datetime.now(tz8).isoformat()

def _count_images(job_folder: Path) -> int:
    """Recursively count images under `job_folder` with common extensions."""
    # Count image files recursively with common extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return sum(1 for p in job_folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)

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
    # Map each endpoint URL to its logical name (used for per-model metrics).
    url_to_name: Dict[str, str] = {client.url: client.name for client in enabled_clients}

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

    # Domain services (pure logic; no I/O, HTTP, or FastAPI dependency).
    merger = CsvMerger()
    classifier = DefectClassifier(get_app_config().defect_rules)
    derived_calc = DerivedMetricsCalculator()

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
            df_processing = merger.add_image_name_column(df_processing)

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

    # Call all enabled model clients concurrently via the infrastructure layer.
    model_results = await run_enabled_model_clients(
        get_app_config(), job_folder, logger=log, req_id=req_id
    )
    results_by_name = {result.name: result for result in model_results}
    for client_cfg in enabled_clients:
        result = results_by_name.get(client_cfg.name)
        if result is None:
            continue
        metrics_by_endpoint[client_cfg.url] = {
            "request_ms": result.request_ms,
            "inference_ms": result.inference_ms,
            "model_version": result.model_version,
            "device": result.device,
        }
        if result.error:
            errors.append(result.error)

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
        
        df_processing = merger.add_image_name_column(df_processing)
        df_processing = merger.merge_model_results(df_processing, model_results)

        # Compute derived metrics (pad_area / cover%) when their inputs exist.
        derived = derived_calc.add_derived_columns(df_processing)
        df_processing = derived.df
        for warning in derived.warnings:
            log.info("event=derived.skip req_id=%s detail=%s", req_id or "-", warning)

        # Rule-based defect classification and pass/fail decision.
        df_processing = classifier.classify(df_processing)
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
