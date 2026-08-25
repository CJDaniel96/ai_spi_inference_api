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

from app.infrastructure.input.sinic_folder_input import (
    FileSettleTracker,
    SinicFolderInput,
)


def setup_logging(base_dir: Path) -> logging.Logger:
    log_dir = base_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scan_jobs.log"

    logger = logging.getLogger("scanner")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if reinitialized
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    fh = TimedRotatingFileHandler(
        str(log_file), when="midnight", backupCount=7, encoding="utf-8"
    )
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
    missing = [
        k
        for k in ("watch_root", "process_api_url", "processed_registry_path")
        if k not in cfg
    ]
    if missing:
        msg = f"ERROR config missing required keys: {', '.join(missing)}"
        (logger.error(msg) if logger else print(msg))
        sys.exit(1)

    # Default interval
    if "rescan_interval_ms" not in cfg or not isinstance(
        cfg.get("rescan_interval_ms"), int
    ):
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


def http_post_with_curl(
    url: str, payload: dict, timeout_sec: float = 30.0
) -> tuple[int, str]:
    # Use curl.exe to POST JSON and emit only HTTP status code
    body = json.dumps(payload, ensure_ascii=False)
    cmd = [
        "curl.exe",
        "-s",  # silent
        "-o",
        "NUL",  # discard body output (Windows null device)
        "-w",
        "%{http_code}",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        body,
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


def process_http_timeout_seconds(cfg: dict) -> float:
    """Return scanner wait time: production deadline plus transport grace."""
    reliability = cfg.get("reliability", {})
    deadline = float(reliability.get("primary_return_deadline_seconds", 30.0))
    grace = float(reliability.get("scanner_http_timeout_grace_seconds", 5.0))
    return max(1.0, deadline + grace)


def http_get_status_with_curl(url: str, timeout_sec: int = 5) -> tuple[int, str]:
    """Use curl.exe to GET and return HTTP status code and stderr.

    Returns (status_code, err). On failure, status_code is -1.
    """
    cmd = [
        "curl.exe",
        "-s",
        "-o",
        "NUL",
        "-w",
        "%{http_code}",
        "-X",
        "GET",
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


# --- Retry / backoff / dead-letter policy for failed job posts ---------------

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_CLIENT_ERROR_MAX_ATTEMPTS = 2
_DEFAULT_BACKOFF_BASE_SECONDS = 5.0
_DEFAULT_BACKOFF_MAX_SECONDS = 300.0


def classify_status(status: int) -> str:
    """Classify an HTTP status (or -1 transport error) for retry decisions.

    Returns "success" (2xx), "client_error" (4xx; will not succeed on retry), or
    "server_error" (5xx or transport failure; may be transient).
    """
    if 200 <= status <= 299:
        return "success"
    if 400 <= status <= 499:
        return "client_error"
    return "server_error"


def backoff_seconds(attempts: int, base: float, cap: float) -> float:
    """Exponential backoff ``base * 2**(attempts-1)`` capped at ``cap``."""
    if attempts <= 0:
        return 0.0
    return min(cap, base * (2.0 ** (attempts - 1)))


class FailureTracker:
    """In-memory retry/backoff/dead-letter state for failing job posts.

    Prevents a persistently-failing "poison" job from being re-posted on every
    scan tick. Client errors (4xx) are dead-lettered quickly (a retry cannot
    fix them); server/transport errors get exponential backoff up to a retry
    cap, after which the job is dead-lettered. State is per-process: a scanner
    restart re-attempts dead-lettered jobs.
    """

    def __init__(
        self,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        client_error_max_attempts: int = _DEFAULT_CLIENT_ERROR_MAX_ATTEMPTS,
        base_backoff_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
        max_backoff_seconds: float = _DEFAULT_BACKOFF_MAX_SECONDS,
    ) -> None:
        self._max_retries = max_retries
        self._client_error_max_attempts = client_error_max_attempts
        self._base = base_backoff_seconds
        self._cap = max_backoff_seconds
        self._state: dict = {}  # key -> {"attempts", "next_retry_at", "dead"}

    def should_skip(self, key: str, now: float) -> tuple:
        """Return ``(skip, reason)`` for a job key at time ``now``."""
        entry = self._state.get(key)
        if entry is None:
            return False, ""
        if entry["dead"]:
            return True, "dead"
        if now < entry["next_retry_at"]:
            return True, "backoff"
        return False, ""

    def record_success(self, key: str) -> None:
        """Clear any failure state after a successful post."""
        self._state.pop(key, None)

    def record_failure(self, key: str, status: int, now: float) -> tuple:
        """Record a failed post; return ``(attempts, dead)``.

        ``dead`` is True when the job has been dead-lettered (give up).
        """
        status_class = classify_status(status)
        entry = self._state.setdefault(
            key, {"attempts": 0, "next_retry_at": 0.0, "dead": False}
        )
        entry["attempts"] += 1
        attempts = entry["attempts"]
        client_capped = (
            status_class == "client_error"
            and attempts >= self._client_error_max_attempts
        )
        if client_capped or attempts >= self._max_retries:
            entry["dead"] = True
            return attempts, True
        entry["next_retry_at"] = now + backoff_seconds(attempts, self._base, self._cap)
        return attempts, False

    def is_dead(self, key: str) -> bool:
        """Return whether ``key`` has been dead-lettered."""
        entry = self._state.get(key)
        return bool(entry and entry["dead"])


def run_sinic_timestamp_scanner(
    *,
    cfg: dict,
    watch_root: Path,
    process_api_url: str,
    registry_path: Path,
    rescan_interval: float,
    logger: logging.Logger,
    retry_tracker: FailureTracker,
) -> int:
    """Watch SINIC ``YYYYMMDDHHMMSS`` folders without requiring a done file."""
    processing = cfg.get("processing", {})
    image_extensions = processing.get("image_extensions", [".jpg", ".jpeg"])
    settle_seconds = float(cfg.get("scanner_settle_seconds", 2.0))
    allow_no_images = bool(cfg.get("scanner_allow_no_images", False))
    adapter = SinicFolderInput(image_extensions)
    settle_tracker = FileSettleTracker(
        settle_seconds=settle_seconds,
        allow_no_images=allow_no_images,
    )
    process_timeout = process_http_timeout_seconds(cfg)

    logger.info("Scanner input mode: sinic_timestamp")
    logger.info("Settle window: %.3fs", settle_seconds)
    logger.info("Process response timeout: %.3fs", process_timeout)
    while True:
        today_date = dt.date.today()
        today_key = today_date.strftime("%Y-%m-%d")
        registry = load_registry(registry_path, logger)
        processed_for_day = set(registry.get(today_key, []))

        for job_dir in adapter.list_candidates(watch_root, today_date):
            job_id = job_dir.name
            if job_id in processed_for_day:
                continue

            job = adapter.load(job_dir)
            if not settle_tracker.observe(job):
                continue

            key = f"{today_key}/{job_id}"
            skip, _reason = retry_tracker.should_skip(key, time.time())
            if skip:
                continue

            payload = {"job_folder": str(job_dir).replace("\\", "/")}
            logger.info(
                "Processing SINIC job %s (%d CSV, %d image) ...",
                job_id,
                len(job.csv_files),
                len(job.image_files),
            )
            status, err = http_post_with_curl(
                process_api_url, payload, timeout_sec=process_timeout
            )
            if 200 <= status <= 299:
                retry_tracker.record_success(key)
                registry.setdefault(today_key, []).append(job_id)
                try:
                    save_registry(registry_path, registry)
                    settle_tracker.forget(job_dir)
                    logger.info("SUCCESS %s -> %d", key, status)
                except Exception as exc:
                    logger.error("ERROR saving registry: %s", exc)
                continue

            detail = err if status == -1 else f"HTTP {status}"
            attempts, dead = retry_tracker.record_failure(key, status, time.time())
            if dead:
                logger.error(
                    "DEAD-LETTER %s after %d attempt(s) (%s); "
                    "giving up until scanner restart",
                    key,
                    attempts,
                    detail,
                )
            else:
                logger.warning(
                    "RETRY %s attempt %d failed (%s); backing off",
                    key,
                    attempts,
                    detail,
                )

        time.sleep(rescan_interval)


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / "config" / "ai_server.json"
    cfg = load_config(config_path)

    watch_root = Path(cfg["watch_root"])  # allow either forward or backslashes
    process_api_url = cfg["process_api_url"]
    rescan_interval = max(100, int(cfg.get("rescan_interval_ms", 1000))) / 1000.0
    registry_path = (
        (root_dir / cfg["processed_registry_path"]).resolve()
        if not os.path.isabs(cfg["processed_registry_path"])
        else Path(cfg["processed_registry_path"]).resolve()
    )

    logger = setup_logging(root_dir)

    retry_tracker = FailureTracker(
        max_retries=int(cfg.get("scanner_max_retries", _DEFAULT_MAX_RETRIES)),
        client_error_max_attempts=int(
            cfg.get(
                "scanner_client_error_max_attempts", _DEFAULT_CLIENT_ERROR_MAX_ATTEMPTS
            )
        ),
        base_backoff_seconds=float(
            cfg.get("scanner_backoff_base_seconds", _DEFAULT_BACKOFF_BASE_SECONDS)
        ),
        max_backoff_seconds=float(
            cfg.get("scanner_backoff_max_seconds", _DEFAULT_BACKOFF_MAX_SECONDS)
        ),
    )

    logger.info("Scanner starting")
    logger.info(f"Watch root: {watch_root}")
    logger.info(f"Process API: {process_api_url}")
    logger.info(f"Registry: {registry_path}")
    logger.info(f"Interval: {rescan_interval:.3f}s")
    process_timeout = process_http_timeout_seconds(cfg)
    logger.info(f"Process response timeout: {process_timeout:.3f}s")

    if cfg.get("scanner_input_mode", "legacy_done") == "sinic_timestamp":
        return run_sinic_timestamp_scanner(
            cfg=cfg,
            watch_root=watch_root,
            process_api_url=process_api_url,
            registry_path=registry_path,
            rescan_interval=rescan_interval,
            logger=logger,
            retry_tracker=retry_tracker,
        )

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

            # Skip poison jobs that are dead-lettered or still in backoff.
            key = f"{today}/{job_id}"
            skip, _reason = retry_tracker.should_skip(key, time.time())
            if skip:
                continue

            job_folder_payload = str(job_dir).replace("\\", "/")
            payload = {"job_folder": job_folder_payload}

            logger.info(f"Processing job {today}/{job_id} ...")
            status, err = http_post_with_curl(
                process_api_url, payload, timeout_sec=process_timeout
            )

            if 200 <= status <= 299:
                # Mark as processed
                retry_tracker.record_success(key)
                registry.setdefault(today, []).append(job_id)
                try:
                    save_registry(registry_path, registry)
                    logger.info(f"SUCCESS {today}/{job_id} -> {status}")
                except Exception as e:
                    logger.error(f"ERROR saving registry: {e}")
            else:
                detail = err if status == -1 else f"HTTP {status}"
                attempts, dead = retry_tracker.record_failure(key, status, time.time())
                if dead:
                    logger.error(
                        f"DEAD-LETTER {today}/{job_id} after {attempts} attempt(s) "
                        f"({detail}); giving up until scanner restart"
                    )
                else:
                    logger.warning(
                        f"RETRY {today}/{job_id} attempt {attempts} failed "
                        f"({detail}); backing off"
                    )

        time.sleep(rescan_interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger = setup_logging(Path(__file__).resolve().parent)
        logger.info("Scanner stopped by user")
        raise SystemExit(0)
