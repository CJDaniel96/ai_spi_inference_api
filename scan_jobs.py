import json
import os
import sys
import time
import datetime as dt
import subprocess
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(base_dir: Path) -> logging.Logger:
    log_dir = base_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scan_jobs.log"

    logger = logging.getLogger("scanner")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if reinitialized
    if logger.handlers:
        return logger

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')

    fh = TimedRotatingFileHandler(str(log_file), when="midnight", backupCount=7, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def load_config(config_path: Path, logger: Optional[logging.Logger] = None) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        msg = f"ERROR reading config '{config_path}': {e}"
        (logger.error(msg) if logger else print(msg))
        sys.exit(1)

    # Validate required keys
    missing = [k for k in ("watch_root", "process_api_url", "processed_registry_path") if k not in cfg]
    if missing:
        msg = f"ERROR config missing required keys: {', '.join(missing)}"
        (logger.error(msg) if logger else print(msg))
        sys.exit(1)

    # Default interval
    if "rescan_interval_ms" not in cfg or not isinstance(cfg.get("rescan_interval_ms"), int):
        cfg["rescan_interval_ms"] = 1000

    return cfg


def load_registry(registry_path: Path, logger: Optional[logging.Logger] = None) -> dict:
    if not registry_path.exists():
        return {}
    try:
        with registry_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    msg = f"WARN registry invalid; starting empty: {registry_path}"
    (logger.warning(msg) if logger else print(msg))
    return {}


def save_registry(registry_path: Path, registry: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def http_post_with_curl(url: str, payload: dict, timeout_sec: int = 30) -> tuple[int, str]:
    # Use curl.exe to POST JSON and emit only HTTP status code
    body = json.dumps(payload, ensure_ascii=False)
    cmd = [
        "curl.exe",
        "-s",            # silent
        "-o", "NUL",    # discard body output (Windows null device)
        "-w", "%{http_code}",
        "-X", "POST",
        url,
        "-H", "Content-Type: application/json",
        "-d", body,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        stdout = (res.stdout or "").strip()
        # curl returns 0 even on HTTP errors; evaluate status from output
        status = int(stdout) if stdout.isdigit() else -1
        return status, res.stderr.strip() if res.stderr else ""
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except FileNotFoundError:
        return -1, "curl.exe not found"
    except Exception as e:
        return -1, str(e)


def http_get_status_with_curl(url: str, timeout_sec: int = 5) -> tuple[int, str]:
    """Use curl.exe to GET and return HTTP status code and stderr.

    Returns (status_code, err). On failure, status_code is -1.
    """
    cmd = [
        "curl.exe",
        "-s",
        "-o", "NUL",
        "-w", "%{http_code}",
        "-X", "GET",
        url,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        stdout = (res.stdout or "").strip()
        status = int(stdout) if stdout.isdigit() else -1
        return status, res.stderr.strip() if res.stderr else ""
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except FileNotFoundError:
        return -1, "curl.exe not found"
    except Exception as e:
        return -1, str(e)


def log_health_checks(logger: logging.Logger) -> None:
    """Check four service health endpoints and log their status codes."""
    endpoints = [
        ("http://localhost:8000/health", "anomaly_score"),
        # ("http://127.0.0.1:8001/health", "paste_pixels"),
        ("http://127.0.0.1:8002/health", "min_pad_distance"),
        ("http://127.0.0.1:5050/health", "5050"),
    ]
    for url, name in endpoints:
        status, err = http_get_status_with_curl(url)
        if 200 <= status <= 299:
            logger.info(f"Health OK [{name}] {url} -> {status}")
        else:
            detail = f" ({err})" if err else ""
            logger.warning(f"Health FAIL [{name}] {url} -> {status}{detail}")


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / "config" / "ai_server.json"
    cfg = load_config(config_path)

    watch_root = Path(cfg["watch_root"])  # allow either forward or backslashes
    process_api_url = cfg["process_api_url"]
    rescan_interval = max(100, int(cfg.get("rescan_interval_ms", 1000))) / 1000.0
    registry_path = (root_dir / cfg["processed_registry_path"]).resolve() if not os.path.isabs(cfg["processed_registry_path"]) else Path(cfg["processed_registry_path"]).resolve()

    logger = setup_logging(root_dir)

    logger.info("Scanner starting")
    logger.info(f"Watch root: {watch_root}")
    logger.info(f"Process API: {process_api_url}")
    logger.info(f"Registry: {registry_path}")
    logger.info(f"Interval: {rescan_interval:.3f}s")

    while True:
        today = dt.date.today().strftime("%Y-%m-%d")
        date_dir = watch_root / today

        registry = load_registry(registry_path, logger)
        processed_for_day = set(registry.get(today, []))

        if not date_dir.exists() or not date_dir.is_dir():
            # Nothing to do until the date folder appears
            time.sleep(rescan_interval)
            continue

        try:
            job_dirs = [p for p in date_dir.iterdir() if p.is_dir()]
        except Exception as e:
            logger.warning(f"cannot list '{date_dir}': {e}")
            time.sleep(rescan_interval)
            continue

        for job_dir in sorted(job_dirs):
            job_id = job_dir.name
            done_file = job_dir / f"{job_id}.done"

            if job_id in processed_for_day:
                continue

            if not done_file.exists():
                # Not ready yet
                continue

            job_folder_payload = str(job_dir).replace("\\", "/")
            payload = {"job_folder": job_folder_payload}

            # Log health of dependent services prior to posting the job
            log_health_checks(logger)

            logger.info(f"Processing job {today}/{job_id} ...")
            status, err = http_post_with_curl(process_api_url, payload)

            if 200 <= status <= 299:
                # Mark as processed
                registry.setdefault(today, []).append(job_id)
                try:
                    save_registry(registry_path, registry)
                    logger.info(f"SUCCESS {today}/{job_id} -> {status}")
                except Exception as e:
                    logger.error(f"ERROR saving registry: {e}")
            else:
                if status == -1:
                    logger.error(f"ERROR calling API: {err}")
                else:
                    logger.error(f"ERROR HTTP {status} for {today}/{job_id}")

        time.sleep(rescan_interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger = setup_logging(Path(__file__).resolve().parent)
        logger.info("Scanner stopped by user")
        raise SystemExit(0)
