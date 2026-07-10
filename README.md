# AI SPI Inference API

A FastAPI service that enriches SPI (Solder Paste Inspection) machine output with
AI model results. It scans a job folder, calls AI inference endpoints in parallel,
merges their scalar results into the CSV(s), applies rule-based defect
classification, decides pass/fail, and writes the enriched CSVs plus a per-job
metrics row.

## What This Project Does

Given a `job_folder` containing SPI CSV(s) and pad images, the service:

1. Validates the folder and discovers its CSV files and image count.
2. Calls the **enabled** model inference endpoints concurrently.
3. Merges each model's scalar result into the CSV by image name.
4. Computes derived columns (`pad_area`, `cover%`) when their inputs exist.
5. Classifies each row's `ai_defect_name` using ordered, config-driven rules.
6. Sets `is_pass` (22 = pass, 23 = fail).
7. Writes a **primary**, **backup**, and **processed** CSV, and appends a metrics
   row to `log/log.csv`.

## Architecture

Modular monolith with a layered architecture under `app/`:

```
app/
  api/            # FastAPI routes + request/response schemas (interface layer)
  application/    # ProcessJobUseCase (orchestration)
  domain/         # pure business logic: entities + services (no I/O)
    entities/     #   Job, ModelInferenceResult, DefectLabel/codes
    services/     #   CsvMerger, DerivedMetricsCalculator, DefectClassifier,
                  #   MetricsCollector, add_is_pass
  infrastructure/ # adapters: model HTTP clients, repositories, output writers
  core/           # config (Pydantic), logging, error hierarchy
  utils/          # small shared helpers
```

The system also involves separate processes:

| Component | Port | Role |
| --- | --- | --- |
| Merge server (`app.main`) | `5050` | This API — orchestration (`/process`, `/health`) |
| PatchCore anomaly | `8000` | Returns `anomaly_score` |
| Paste detection | `8001` | Returns `paste_pixels` — **disabled by default** |
| Distance detection | `8002` | Returns `min_center_to_pad_distance` |
| `scan_jobs.py` | — | Folder watcher that POSTs ready jobs to `/process` |

**Entry points**

- Primary (new architecture): `python -m app.main` — runs entirely on the layered
  `app/` code.
- Legacy (deprecated, kept for compatibility): `python ai_server_fastapi.py`.

## Runtime Flow

```
scan_jobs.py  ──POST /process {job_folder}──▶  app.main
    │                                              │
    │                              ProcessJobUseCase.execute()
    │        1. FileSystemJobRepository.load  (validate, find CSVs, count images)
    │        2. if images > threshold → skip inference, mark all is_pass=23
    │        3. run enabled model clients concurrently  (asyncio.gather)
    │        4. per CSV: merge → derived → classify → is_pass
    │        5. OutputWriter: primary + backup + processed CSV
    │        6. RequestLogWriter: append metrics row to log/log.csv
    ▼
2xx → scan_jobs marks the job processed
```

## API Reference

### GET /health

Cheap liveness probe — returns immediately with the current UTC+8 timestamp.

```json
{ "status": "healthy", "time": "2026-07-09T10:00:00+08:00" }
```

### GET /ready

Readiness probe. Returns **200** when the config loads (the service can accept
jobs) or **503** when it cannot. Each enabled model endpoint's `/health` is probed
(short timeout) and reported for diagnostics, but a model being down does **not**
flip readiness (since `/process` tolerates a single model failure).

```json
{
  "status": "ready",
  "config_ok": true,
  "config_error": null,
  "models": [
    { "name": "anomaly",  "url": "http://127.0.0.1:8000/health", "healthy": true,  "detail": "status=200" },
    { "name": "distance", "url": "http://127.0.0.1:8002/health", "healthy": false, "detail": "ConnectError" }
  ]
}
```

### POST /process

Body:

```json
{ "job_folder": "D:/path/to/job" }
```

Success (200):

```json
{
  "status": "ok",
  "saved_files": [".../AI/xxx.csv", ".../backup/.../xxx.csv", ".../xxx_processed.csv"],
  "errors": [],
  "csv_count": 1
}
```

Skipped (200, image count over threshold):

```json
{
  "status": "finished scanning",
  "skipped": true,
  "reason": "images_exceed_threshold",
  "img_numbers": 812,
  "csv_count": 1,
  "saved_files": ["..."],
  "errors": []
}
```

Errors:

| Situation | HTTP | Body |
| --- | --- | --- |
| Job folder missing / not a directory | 400 | `{ "detail": "..." }` |
| No CSV files in folder | 400 | `{ "detail": "..." }` |
| CSV missing required columns / bad values | 400 | `{ "detail": "..." }` |
| A single model endpoint fails | 200 | success body with the message in `errors` |
| Config invalid / output write fails / unexpected | 500 | `{ "detail": "Internal Server Error" }` |

`curl` example:

```bash
curl -X POST "http://127.0.0.1:5050/process" \
  -H "Content-Type: application/json" \
  -d '{ "job_folder": "D:/Dre/JQ_SPI_02_AI_API/data/20250523135357" }'
```

## Input Job Folder Requirements

- A directory that exists and contains **one or more `.csv` files**.
- Pad images named `{Array_id - 1}_{Pad_no}.jpg` (used to align model results to
  rows). Image extensions counted for the guardrail come from
  `processing.image_extensions`.

## CSV Required Columns

- **Required** (to build the `img_name` join key): `Array_id`, `Pad_no`.
  Missing either raises a CSV schema error (HTTP 400).
- **Optional**, consulted by defect rules only when present:
  `insp_vol`, `vol_l_ng`, `vol_h_ng`, `insp_height`, `Width`, `Length`.
- **Produced by models** (added during merge): `anomaly_score`,
  `min_pad_distance`, and `paste_pixels` (only if the paste model is enabled).
  Missing model columns become `NaN` — rules that need them are simply skipped.

## Model Client Configuration

Model endpoints are **not** hard-coded. The service calls only the enabled entries
in `model_clients`:

```json
{
  "name": "anomaly",
  "enabled": true,
  "url": "http://127.0.0.1:8000/inference",
  "target_column": "anomaly_score",
  "timeout_seconds": 300
}
```

- **Disable** a client: set `"enabled": false` (its column is left `NaN`).
- **Add** a client: append an object with a unique `name`, `url`, `target_column`,
  and `timeout_seconds`, then set `"enabled": true`. No code change required.
- The paste client (`8001`) ships **disabled** (see compatibility notes).

## Defect Classification Rules

Applied in priority order; only the **first** matching rule per row assigns a
label. Thresholds/offsets come from `defect_rules`.

| # | Condition | Label |
| --- | --- | --- |
| 1 | `anomaly_score > anomaly_threshold` | `FM/color` |
| 2 | `insp_vol > vol_h_ng + high_vol_offset` | `high vol` |
| 3 | `insp_vol < vol_l_ng + low_vol_offset` | `low vol` |
| 4 | `cover% > high_cover_threshold` | `high cover` |
| 5 | `min_pad_distance < short_distance_threshold` | `short distance` |
| 6 | `insp_height > high_paste_height_threshold` | `high paste` |

`is_pass` = `22` when `ai_defect_name` is empty/whitespace, else `23`.

Note rule 5 uses **`<` (less than)** the threshold.

## Output Files

For each input CSV, three files are written:

| File | Path | Contents |
| --- | --- | --- |
| Primary | `{external_output_root}/{job}/AI/{csv}` | Per `primary_csv_mode` (below) |
| Backup | `{backup_output_root}/{year}/{month}/{job}/{csv}` | Original columns + `is_pass` |
| Processed | `{backup_output_root}/{year}/{month}/{job}/{stem}_processed.csv` | Full AI schema |

`output.primary_csv_mode`:

- `is_pass_only` (default): the original columns with **only `is_pass` updated**
  (original numeric formatting preserved).
- `full_ai_columns`: the full processed frame, including `img_name`, model score
  columns, `ai_defect_name`, and `is_pass`.

The backup and `_processed.csv` are **unchanged by the mode**. The
`*_processed.csv` carries the full AI schema and is intended for **debugging**.

The per-job metrics row (`log/log.csv`) keeps its 18-column schema:
`job_folder, img_numbers, request_start_at, request_end_at, anomaly_request_ms,
paste_request_ms, distance_request_ms, compute_total_ms, request_latency_ms,
pass_count, fail_count, anomaly_count, distance_count, low_vol_count,
high_vol_count, high_cover_count, high_paste_count, logged_at`.

## Config Reference

Single file: `config/ai_server.json` (override the path with the `AI_CONFIG_PATH`
environment variable). Validated on load by `app/core/config.py`.
`config/ai_server.example.json` is a committed template with placeholder paths —
copy it (or point `AI_CONFIG_PATH` at a deployment-specific file) rather than
baking environment-specific paths into a shared file.

```json
{
  "server": { "host": "127.0.0.1", "port": 5050 },
  "paths": {
    "external_output_root": "D:/Dre/JQ_SPI_02_AI_API/data",
    "backup_output_root": "D:/Dre/JQ_SPI_02_AI_API/backup"
  },
  "processing": {
    "folder_images_num_threshold": 500,
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
  },
  "model_clients": [
    { "name": "anomaly",  "enabled": true,  "url": "http://127.0.0.1:8000/inference", "target_column": "anomaly_score",    "timeout_seconds": 300 },
    { "name": "paste",    "enabled": false, "url": "http://127.0.0.1:8001/inference", "target_column": "paste_pixels",     "timeout_seconds": 300 },
    { "name": "distance", "enabled": true,  "url": "http://127.0.0.1:8002/inference", "target_column": "min_pad_distance", "timeout_seconds": 300 }
  ],
  "defect_rules": {
    "anomaly_threshold": 0.9,
    "high_cover_threshold": 180.0,
    "short_distance_threshold": 6.8,
    "low_vol_offset": -10.0,
    "high_vol_offset": 20.0,
    "high_paste_height_threshold": 200.0
  },
  "output": { "primary_csv_mode": "is_pass_only" },
  "logging": {
    "log_dir": "log", "system_log_file": "system", "request_log_file": "log.csv",
    "request_log_max_bytes": 52428800, "request_log_backup_count": 5
  },

  "watch_root": "D:/spi_ai/output/01/sfcTemp",
  "process_api_url": "http://127.0.0.1:5050/process",
  "processed_registry_path": "log/processed.json",
  "rescan_interval_ms": 500
}
```

The top-level `watch_root` / `process_api_url` / `processed_registry_path` /
`rescan_interval_ms` keys are read directly by `scan_jobs.py`.

Config is cached per process — **restart the server to pick up config changes**
(the scanner re-reads its own config each loop).

## How to Install

The project uses [`uv`](https://docs.astral.sh/uv/). `pyproject.toml` is the
canonical dependency source; `requirements-*.txt` are pinned mirrors for non-uv
installs. Python `>=3.10,<3.12`.

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows/Linux + CUDA
uv venv --python 3.10
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv sync --extra cuda --group dev

# macOS (CPU / Apple Silicon) — no CUDA-only packages
uv venv --python 3.10
uv sync --extra mac --group dev
```

CUDA-only packages (`pycuda`, `onnxruntime-gpu`, TensorRT) never install on macOS.
See the extras in `pyproject.toml` (`cuda`, `mac`, `analytics`, `paste`).

## How to Run

```bash
# The merge API (this service)
uv run python -m app.main            # binds server.host:port (default 127.0.0.1:5050)

# Model servers + scanner (Windows) — starts 5050, 8000, 8002, scanner
02_api_services.bat
```

The legacy entry point `uv run python ai_server_fastapi.py` is still available but
deprecated.

## Deployment / Operations

- **Auto-restart**: `02_api_services.bat` launches the merge server via
  `run_merge_server.bat`, which relaunches `python -m app.main` if it exits or
  crashes. For a proper Windows service (auto-start on boot, crash recovery,
  graceful stop), install it under **NSSM** instead:
  `nssm install SPI_Merge <PYTHON_EXE> -m app.main` (set the working directory to
  the repo root).
- **Graceful shutdown**: the app manages a shared HTTP client via a FastAPI
  lifespan; on SIGINT/SIGTERM (uvicorn) it finishes in-flight work and closes the
  client cleanly.
- **Metrics log rotation**: `log/log.csv` rotates at
  `logging.request_log_max_bytes` (default 50 MB) keeping
  `logging.request_log_backup_count` backups (`log.csv.1..N`); set
  `request_log_max_bytes: 0` to disable. No data is lost until the backup count is
  exceeded.
- **Config per environment**: keep deployment paths out of the shared repo by
  pointing `AI_CONFIG_PATH` at a machine-local config (template:
  `config/ai_server.example.json`).
- **CI**: `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, and
  the `tests/unit` + `tests/integration` suites on every push/PR.

## How to Test

```bash
uv run pytest
```

Unit tests (`tests/unit/`) cover the pure domain services and config; integration
tests (`tests/integration/`) drive `ProcessJobUseCase` and the `/process` route
using temporary directories and a fake model runner — **no real endpoints or
production paths are required**.

## How to Format and Lint

The project uses **Ruff** for both linting and formatting (no Black — a single
formatter avoids conflicts):

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ruff format --check . # verify formatting in CI
uv run mypy app              # optional type check
```

Legacy top-level scripts (`ai_server_fastapi.py`, the model servers, `scan_jobs.py`,
etc.) are listed in `extend-exclude` in `pyproject.toml` and are intentionally not
linted/formatted yet — see "Known Behavior".

## Troubleshooting

- **Empty model columns**: ensure model responses key `results` by the same
  `img_name` format (`{Array_id-1}_{Pad_no}.jpg`).
- **`400 CSV missing required columns`**: the CSV needs `Array_id` and `Pad_no`.
- **`high cover` never fires**: `cover%` needs `paste_pixels` (paste model) and
  `pad_area` (needs `Width` + `Length`). Paste is disabled by default.
- **Writes fail / `500`**: check `external_output_root` / `backup_output_root`
  exist and are writable; output-write failures return a 500.
- **Config errors on startup**: `config/ai_server.json` must be present and valid
  (or set `AI_CONFIG_PATH`).

## Known Behavior and Compatibility Notes

- **Paste model (8001) is disabled by default.** It is a heavier YOLO + MobileSAM
  pipeline and the `cover%` / `high cover` rule depends on it. To enable: install
  the `paste` extra, start the paste server on `8001`, set the `paste` client's
  `"enabled": true`, and restart the merge server.
- **`short distance` uses `min_pad_distance < short_distance_threshold`** (less
  than), not greater than.
- **`primary_csv_mode`** has two modes: `is_pass_only` (default) and
  `full_ai_columns` (see Output Files). Backup and processed CSVs are unaffected.
- **`*_processed.csv`** contains the full AI schema and exists for **debugging**.
- **Image count over `folder_images_num_threshold`**: inference is **skipped**,
  every row is marked `is_pass = 23` (fail), the three CSVs are still written, and
  the response is `{"status": "finished scanning", "skipped": true, "reason":
  "images_exceed_threshold"}`.
- **A single model endpoint failure** does not fail the job: its message is added
  to the response `errors` array and processing continues (HTTP 200).
- **`log/`, `backup/`, `data/`** and model weights are git-ignored.
- **Binds `127.0.0.1` by default** (`server.host`): the only client is the
  same-host scanner. Set `server.host` to a wider interface only if needed.
- **`/process` offloads CPU/IO-bound work** (CSV read/write, classification) to a
  thread pool, so `/health` and `/ready` stay responsive while a job runs.
- **Output CSVs are written atomically** (temp file + `os.replace`), so a crash
  mid-write never leaves a partial CSV at the target path.
- **Legacy code**: `ai_server_fastapi.py` (and the model servers / scanner) remain
  for the legacy entry point and are excluded from Ruff/mypy. The new `app/` entry
  point has no runtime dependency on `ai_server_fastapi.py`.
