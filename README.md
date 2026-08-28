# AI SPI Inference API

A durable three-stage pipeline that enriches SPI (Solder Paste Inspection) machine
output with AI results and returns the machine-facing CSV within a configurable
soft-real-time deadline. Ingest, inference, and publication run as independent
processes coordinated through a local SQLite database. The original synchronous
FastAPI `/process` workflow remains available for compatibility.

## What This Project Does

Given a timestamp folder containing SPI CSV(s) and pad images, the durable pipeline:

1. Watches today's 14-digit timestamp folders and waits for a stable file set.
2. Parses every CSV and verifies that every expected image exists, is non-empty,
   and can be decoded.
3. Persists `ready_at` and the absolute deadline in SQLite, then atomically copies
   the complete original tree to the AIPC's local backup.
4. Calls the **enabled** model endpoints against the immutable local copy.
5. Verifies that every required model returned a finite result for every expected
   image. Required-model failure or timeout produces an all-`23` fail-safe result.
6. Merges results, calculates derived values, classifies defects, and writes a
   durable result manifest without publishing to the machine share.
7. Publishes every Primary CSV first, then writes the returned/processed result
   backups and marks the job complete.

## Architecture

Layered application with independently runnable workers under `app/`:

```
app/
  api/                     # FastAPI compatibility routes and schemas
  application/
    services/              # ingest, inference-artifact, and publisher services
    workers/               # three independent durable worker loops
    use_cases/             # legacy synchronous ProcessJobUseCase
  domain/                  # jobs, pipeline states/results, classification logic
  infrastructure/
    repositories/          # CSV/filesystem adapters and SQLite job repository
    model_clients/         # deadline-aware HTTP model clients
    output/                # atomic machine-facing CSV output
  core/                    # Pydantic config, logging, and error hierarchy
  pipeline.py              # `python -m app.pipeline <stage>` CLI
```

Production components:

| Component | Port | Role |
| --- | --- | --- |
| Ingest worker | — | Validate today's timestamp folders and create local immutable input |
| Inference worker | — | Earliest-deadline-first model execution and result manifest creation |
| Publisher worker | — | Deadline takeover, Primary-first publication, then local result backup |
| SQLite WAL database | — | Durable job state, leases, attempts, timestamps, and crash recovery |
| PatchCore anomaly | `8000` | Returns `anomaly_score` |
| Paste detection | `8001` | Returns `paste_pixels` — **disabled by default** |
| Distance detection | `8002` | Returns `min_center_to_pad_distance` |
| Merge server (`app.main`) | `5050` | Compatibility API (`/process`, `/health`, `/ready`) |

**Entry points**

- Durable production pipeline: `python -m app.pipeline ingest`, `inference`, and
  `publisher` in three separate processes.
- Compatibility API: `python -m app.main`.
- Deprecated legacy server: `python ai_server_fastapi.py`.

## Runtime Flow

```text
SPI shared folder
       │
       ▼
Ingest Worker
  stable snapshot + CSV/image validation
  persist INGESTING + ready_at/deadline_at
  atomic complete local copy
       │ READY
       ▼
Inference Worker
  earliest deadline first
  required models within deadline - publish reserve
  normal decision or all-23 fallback manifest
       │ RESULT_READY / FALLBACK_READY
       ▼
Publisher Worker
  normal result, or cutoff takeover from INGESTING/READY/INFERENCING
  wait for the complete durable raw copy if ingest is still finishing
  publish all Primary CSVs atomically
       │ PRIMARY_RETURNED
       ▼
  local returned CSV + processed CSV + manifest backup
       │ DONE
       ▼
SQLite WAL state and stage-specific logs retain the evidence
```

Persisted states are `INGESTING`, `READY`, `INFERENCING`, `RESULT_READY`,
`FALLBACK_READY`, `PUBLISHING`, `PRIMARY_RETURNED`, `DONE`, and `FAILED`.
Claims use SQLite transactions and expiring worker leases, so an expired stage can
be reclaimed after a process crash. Jobs are ordered by the earliest deadline.

### 30-second soft-real-time policy

`ready_at` is recorded after the timestamp folder has a stable snapshot and the
complete CSV/image contract has passed validation. The absolute deadline is:

```text
deadline_at = ready_at + primary_return_deadline_seconds
publish_cutoff = deadline_at - primary_publish_reserve_seconds
```

With the shipped `30`-second deadline and `5`-second reserve, model work must be
committable before approximately second 25. At the cutoff, Publisher atomically
takes ownership of an unfinished ingest/inference job and selects an all-`23`
fallback. Primary is produced only from the complete local raw copy, never directly
from the machine share; this preserves the original evidence even if the machine
deletes its folder after seeing the return CSV. A late inference commit is rejected
by SQLite and cannot overwrite the fallback. Optional model tasks are cancelled
once all required models have finished, so they do not consume the publication
reserve.

The SLA covers local archival, queueing, inference, result construction, and the
last Primary CSV write after `ready_at`; folder discovery, settle time, and
validation happen before `ready_at`. This is a **soft** real-time target: an AIPC
stall, sustained overload, unavailable/blocked SMB share, or unusually slow raw
copy can still miss it. A slow copy is allowed to miss the soft target rather than
returning before the required original backup exists.
`pipeline.primary_returned` logs include latency and `deadline_met` evidence.

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
flip readiness (`/process` publishes the configured all-23 fail-safe result when
a required model is unavailable).

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

This endpoint intentionally keeps the original synchronous contract. It runs
`ProcessJobUseCase` inline and **does not enqueue or update the three-stage SQLite
pipeline**. It remains useful for existing clients and manual processing, but its
deadline starts when the request begins rather than at the durable pipeline's
`ready_at` timestamp.

Do not run `scan_jobs.py` and the new Ingest Worker against the same watched folders
at the same time: they are two separate execution paths and can publish the same
job concurrently. For production, choose one mode:

- Durable mode: run the three `app.pipeline` workers; the API is optional.
- Compatibility mode: run `scan_jobs.py` plus `app.main`; do not run the new Ingest
  Worker for that share.

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

Fail-safe result (200, required model failure or inference-budget timeout):

```json
{
  "status": "ok",
  "fallback": true,
  "reason": "required_model_failure",
  "img_numbers": 4,
  "csv_count": 1,
  "saved_files": ["..."],
  "errors": ["distance boom"]
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
| Required model fails/times out/returns incomplete data | 200 | all rows are returned as `23` |
| Optional model endpoint fails | 200 | normal result with the message in `errors` |
| Config invalid / output write fails / unexpected | 500 | `{ "detail": "Internal Server Error" }` |

`curl` example:

```bash
curl -X POST "http://127.0.0.1:5050/process" \
  -H "Content-Type: application/json" \
  -d '{ "job_folder": "D:/Dre/JQ_SPI_02_AI_API/data/20250523135357" }'
```

## Input Job Folder Requirements

- A 14-digit timestamp directory (`YYYYMMDDHHMMSS`) for today's date containing
  **one or more `.csv` files**. Automatic processing scans only today; a historical
  date can be submitted manually with
  `app.pipeline ingest --once --date YYYY-MM-DD`.
- SINIC pad images named
  `{Insp_st_time}_{BoardBarcode}_{component_name}_{Array_id}_{Pad_id}.jpg`.
  The shipped template uses the equivalent
  `{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg`.
- Every CSV must contain at least one data row, and expected image names must be
  unique across every CSV in the job.
- Every expected image must be a regular file, have non-zero size, and be decodable
  by OpenCV. The source snapshot must remain unchanged during validation and copy.

The settle duration comes from `pipeline.source_settle_seconds`. Because the
machine supplies neither a completion file nor an atomic folder rename, settle +
contract validation is the implemented definition of "folder ready".

## CSV Required Columns

- **Required** by the shipped SINIC template: `component_name`, `Array_id`, and
  `Pad_id`; the CSV stem represents `{Insp_st_time}_{BoardBarcode}`.
- **Optional**, consulted by defect rules only when present:
  `insp_vol`, `vol_l_ng`, `vol_h_ng`, `insp_hei`, `Width`, `Length`.
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
  "required": true,
  "url": "http://127.0.0.1:8000/inference",
  "target_column": "anomaly_score",
  "timeout_seconds": 25
}
```

- **Disable** a client: set `"enabled": false` (its column is left `NaN`).
- **Required** client: set `"required": true`; any error, missing image key,
  `None`, `NaN`, or infinite result triggers the all-23 fail-safe output.
- **Optional** client: set `"required": false`; its failure is reported but does
  not trigger the fail-safe policy.
- **Add** a client: append an object with a unique `name`, `url`, `target_column`,
  and `timeout_seconds`, then set `"enabled": true`. No code change required.
- The paste client (`8001`) ships **disabled** (see compatibility notes).

## Defect Classification Rules

Applied in the order listed by `defect_rules.rule_order`; only the **first**
matching rule per row assigns a label. Remove a name from `rule_order` to disable
that rule, or use an empty list to disable all six rules. Unknown or duplicate
names are rejected when configuration is loaded. Thresholds/offsets also come
from `defect_rules`.

| # | Condition | Label |
| --- | --- | --- |
| 1 | `anomaly_score > anomaly_threshold` | `FM/color` |
| 2 | `insp_vol > vol_h_ng + high_vol_offset` | `high vol` |
| 3 | `insp_vol < vol_l_ng + low_vol_offset` | `low vol` |
| 4 | `cover% > high_cover_threshold` | `high cover` |
| 5 | `min_pad_distance < short_distance_threshold` | `short distance` |
| 6 | `insp_hei > high_paste_height_threshold` | `high paste` |

The corresponding configuration names are `anomaly`, `high_vol`, `low_vol`,
`high_cover`, `short_distance`, and `high_paste`. For example, this evaluates
short distance first and disables anomaly and high-cover classification:

```json
"rule_order": ["short_distance", "high_vol", "low_vol", "high_paste"]
```

Restart every running pipeline worker (and the compatibility API when used) after
changing the config. `rule_order` controls only classification; model execution
and fail-safe requirements remain controlled by the corresponding
`model_clients[].enabled` and `model_clients[].required` settings.

`is_pass` = `22` when `ai_defect_name` is empty/whitespace, else `23`.

Note rule 5 uses **`<` (less than)** the threshold.

## Output Files

### Durable pipeline paths

| File | Path | Contents |
| --- | --- | --- |
| Original source copy | `{backup_output_root}/{YYYY-MM-DD}/{job_id}/...` | Complete byte-verified source payload before inference; source files are never modified |
| Optional staging copy | `{pipeline.staging_root}/{job_id}/...` | Separate local inference copy only when `staging_root` is set |
| Result artifact | `{pipeline.result_root}/{job_id}/{attempt-id}/manifest.json` | Normal/fallback decision, row-aligned result codes, errors, timings, and counts |
| Staged processed CSV | `{pipeline.result_root}/{job_id}/{attempt-id}/processed/{stem}_processed.csv` | Full AI/debug schema used by Publisher |
| Primary | `{external_output_root}/{csv}` | Original CSV format with only `is_pass` replaced; atomic per file |
| Primary with preserved job folder | `{external_output_root}/{job_id}/{csv}` | Used when `output.preserve_job_folder=true` |
| Returned result backup | `{backup_output_root}/{YYYY-MM-DD}/{job_id}/ai_result/returned/{csv}` | Local copy of the machine-facing result |
| Processed result backup | `{backup_output_root}/{YYYY-MM-DD}/{job_id}/ai_result/processed/{stem}_processed.csv` | Full AI/debug schema after Primary is visible |
| Result manifest backup | `{backup_output_root}/{YYYY-MM-DD}/{job_id}/ai_result/manifest.json` | Durable outcome/reason/errors and CSV result codes |

When `pipeline.staging_root` is `null` (the recommended default), the preserved
original source copy is also the inference staging directory, avoiding a second
full copy inside the 30-second budget. When it is set to a different local root,
Ingest creates and verifies two complete atomic copies.

Publisher writes **all Primary CSVs before any** returned/processed backup. A local
result-backup failure leaves the state at `PRIMARY_RETURNED`; a later Publisher
claim retries finalization without rerunning inference or changing the Primary.

When the durable pipeline is enabled, startup validation requires
`primary_csv_mode="is_pass_only"` and `primary_path_layout="machine_return"`.
It uses the machine-return contract implemented by `SinicCsvOutput`: source
encoding, delimiter, column order, quoting, and line endings are preserved while
`is_pass` changes. The compatibility `/process` workflow can use its other modes
only when the durable pipeline is disabled. Publisher also uses
`output.preserve_job_folder` and `output.require_existing_is_pass`.

### Compatibility `/process` paths

The synchronous compatibility path retains its previous output layout:

| File | Path |
| --- | --- |
| Primary (`machine_return`) | `{external_output_root}/{csv}` |
| Primary (legacy layout) | `{external_output_root}/{job}/AI/{csv}` |
| Backup | `{backup_output_root}/{year}/{month}/{job}/{csv}` |
| Processed | `{backup_output_root}/{year}/{month}/{job}/{stem}_processed.csv` |

`output.primary_csv_mode`:

- `is_pass_only` (default): the original columns with **only `is_pass` updated**
  (original numeric formatting preserved).
- `full_ai_columns`: the full processed frame, including `img_name`, model score
  columns, `ai_defect_name`, and `is_pass`.

On the compatibility path, backup and `_processed.csv` are unchanged by the mode.
The `*_processed.csv` carries the full AI schema and is intended for debugging.

The compatibility `/process` per-job metrics row (`log/log.csv`) keeps its
18-column schema:
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
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
    "image_name_source_column": null,
    "image_name_template": "{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg"
  },
  "model_clients": [
    { "name": "anomaly",  "enabled": true,  "required": true,  "url": "http://127.0.0.1:8000/inference", "target_column": "anomaly_score",    "timeout_seconds": 25 },
    { "name": "paste",    "enabled": false, "required": false, "url": "http://127.0.0.1:8001/inference", "target_column": "paste_pixels",     "timeout_seconds": 25 },
    { "name": "distance", "enabled": true,  "required": true,  "url": "http://127.0.0.1:8002/inference", "target_column": "min_pad_distance", "timeout_seconds": 25 }
  ],
  "defect_rules": {
    "rule_order": [
      "anomaly", "high_vol", "low_vol", "high_cover",
      "short_distance", "high_paste"
    ],
    "anomaly_threshold": 0.9,
    "high_cover_threshold": 180.0,
    "short_distance_threshold": 6.8,
    "low_vol_offset": -10.0,
    "high_vol_offset": 20.0,
    "high_paste_height_threshold": 200.0
  },
  "output": {
    "primary_csv_mode": "is_pass_only",
    "primary_path_layout": "machine_return",
    "preserve_job_folder": false,
    "require_existing_is_pass": true
  },
  "reliability": {
    "primary_return_deadline_seconds": 30.0,
    "primary_publish_reserve_seconds": 5.0,
    "scanner_http_timeout_grace_seconds": 5.0,
    "required_model_failure_policy": "fail_all_23"
  },
  "pipeline": {
    "enabled": true,
    "watch_root": "D:/spi_ai/output/01/sfcTemp",
    "database_path": "D:/Dre/JQ_SPI_02_AI_API/state/pipeline.sqlite3",
    "staging_root": null,
    "result_root": "D:/Dre/JQ_SPI_02_AI_API/pipeline_results",
    "source_settle_seconds": 2.0,
    "ingest_poll_interval_seconds": 0.25,
    "inference_poll_interval_seconds": 0.05,
    "publisher_poll_interval_seconds": 0.05,
    "publisher_lease_seconds": 1.0,
    "publisher_heartbeat_interval_seconds": 0.25,
    "worker_lease_seconds": 60.0
  },
  "logging": {
    "log_dir": "log", "system_log_file": "system", "request_log_file": "log.csv",
    "request_log_max_bytes": 52428800, "request_log_backup_count": 5
  },

  "watch_root": "D:/spi_ai/output/01/sfcTemp",
  "scanner_input_mode": "sinic_timestamp",
  "scanner_settle_seconds": 2.0,
  "process_api_url": "http://127.0.0.1:5050/process",
  "processed_registry_path": "log/processed.json",
  "rescan_interval_ms": 500
}
```

Pipeline settings:

| Field | Meaning |
| --- | --- |
| `enabled` | Must be `true` or `app.pipeline` exits at startup |
| `watch_root` | Shared root scanned by the new Ingest Worker; distinct from the legacy top-level key |
| `database_path` | Local file-backed SQLite WAL database; never place it on SMB/network storage |
| `staging_root` | `null` reuses the preserved original copy; otherwise creates `{staging_root}/{job_id}` |
| `result_root` | Local transient result-manifest and processed-artifact root |
| `source_settle_seconds` | Stable snapshot window before full CSV/image validation |
| `*_poll_interval_seconds` | Delay between worker polling iterations |
| `publisher_lease_seconds` | Short Primary claim lease; must fit inside the publish reserve |
| `publisher_heartbeat_interval_seconds` | How often a live Publisher renews its short lease |
| `worker_lease_seconds` | Ingest/inference/finalize lease; must exceed the deadline |

Publisher actively renews its short lease while building/publishing Primary files.
If that process crashes, another Publisher can recover inside the reserved window.
Ingest/inference use the longer `worker_lease_seconds`; deadline takeover still
fences them at the publish cutoff.

The top-level `watch_root`, `scanner_settle_seconds`, `process_api_url`,
`processed_registry_path`, and `rescan_interval_ms` keys are legacy settings read
by `scan_jobs.py`. Durable Ingest reads `pipeline.watch_root` and
`pipeline.source_settle_seconds`.

`primary_return_deadline_seconds` defaults to 30 seconds. The durable pipeline
persists the deadline from validated `ready_at`; the synchronous `/process` path
still measures from the beginning of the HTTP request. `scanner_http_timeout_grace_seconds`
applies only to `scan_jobs.py`.

Config is cached independently in each process. Restart all three workers (and the
API when used) after changing settings. `scan_jobs.py` continues to load its own
legacy configuration.

## How to Install

The project uses [`uv`](https://docs.astral.sh/uv/). `pyproject.toml` is the
canonical dependency source; `requirements-*.txt` are pinned mirrors for non-uv
installs. Python `>=3.12,<3.13`.

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows/Linux + CUDA
uv venv --python 3.12
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv sync --extra cuda --group dev

# macOS (CPU / Apple Silicon) — no CUDA-only packages
uv venv --python 3.12
uv sync --extra mac --group dev
```

CUDA-only packages (`pycuda`, `onnxruntime-gpu`, TensorRT) never install on macOS.
See the extras in `pyproject.toml` (`cuda`, `tensorrt-export`, `mac`,
`analytics`, `paste`).

## TensorRT Model Conversion Tool

Build TensorRT engines on the target NVIDIA AIPC. TensorRT plans are tied to
the GPU architecture and TensorRT/CUDA runtime; do not build on a development
PC and assume the plan is portable. The converter requires TensorRT 10.x except
10.1.x and supports FP32/FP16. INT8 is deliberately excluded until a
representative calibration set and SPI 22/23 decision-parity test are available.

Install the optional converter dependencies in the normal CUDA environment:

```bat
pip install -r requirements-tensorrt-export.txt
```

TensorRT itself is installed separately from NVIDIA so it can match the AIPC
driver/CUDA image. The tool imports converter dependencies lazily and does not
change production worker startup.

### YOLO `.pt` to `.engine`

The Distance service currently pads center/pad inference to a fixed batch of 8,
so the production defaults are FP16, batch 8, and 640x640:

```bat
python convert_to_tensorrt.py yolo ^
  --input models\distance\pad.pt ^
  --output models\distance\pad.engine ^
  --task detect ^
  --precision fp16 ^
  --batch 8 ^
  --imgsz 640 640 ^
  --workspace-gib 4 ^
  --save-onnx models\distance\pad.onnx
```

Add `--test-image D:\path\to\pad.jpg` to run a full padded inference batch
after structural engine verification. The `.pt` route uses the pinned
Ultralytics 8.3.187 exporter to produce ONNX, then uses this project's builder
to create an exact TensorRT profile.

### YOLO `.onnx` to `.engine`

Ultralytics cannot re-export a loaded ONNX model. This route uses TensorRT's
ONNX parser and writes the metadata prefix required by `YOLO(engine)`. An ONNX
exported by Ultralytics normally contains names/task metadata. Otherwise provide
them explicitly. When NMS metadata is also absent, declare the graph's real
output as `raw` or `end2end`; this declaration never rewrites the graph:

```bat
python convert_to_tensorrt.py yolo ^
  --input D:\models\center.onnx ^
  --output models\distance\center.engine ^
  --task detect ^
  --class-names center ^
  --onnx-output-contract raw ^
  --precision fp16 ^
  --batch 8 ^
  --imgsz 640 640
```

For a dynamic ONNX, use an explicit profile. Height/width remain fixed for the
current services:

```bat
python convert_to_tensorrt.py yolo ^
  --input D:\models\center_dynamic.onnx ^
  --output D:\models\center_dynamic.engine ^
  --task detect --class-names center --onnx-output-contract raw ^
  --dynamic --min-batch 1 --batch 4 --max-batch 8 ^
  --imgsz 640 640
```

### PatchCore `.pt`/checkpoint to `.engine`

PatchCore output is normalized to the current runtime contract:

- one `input` tensor in NCHW RGB `[0,1]` format;
- `anomaly_map` and `pred_score` outputs;
- a dynamic batch profile of 1/8/8 by default;
- a raw TensorRT plan without the Ultralytics metadata prefix.

An anomalib-exported `model.pt` contains a Python-pickled module. Only load a
trusted file and explicitly acknowledge that boundary. Standard anomalib 0.7
preprocessing commonly resizes to 256, center-crops to 224, and applies
ImageNet normalization; those operations must match training:

```bat
python convert_to_tensorrt.py patchcore ^
  --input D:\models\patchcore\model.pt ^
  --output models\patchcore\model_fp16.engine ^
  --trust-pickle ^
  --preprocess imagenet ^
  --input-size 256 256 ^
  --center-crop 224 224 ^
  --dynamic --min-batch 1 --batch 8 --max-batch 8 ^
  --precision fp16 ^
  --save-onnx models\patchcore\model_runtime.onnx
```

For a Lightning/raw `state_dict`, architecture cannot be recovered from tensor
weights. Supply the exact training values:

```bat
python convert_to_tensorrt.py patchcore ^
  --input D:\models\patchcore\model.ckpt ^
  --output models\patchcore\model_fp16.engine ^
  --preprocess none ^
  --input-size 256 256 ^
  --backbone efficientnet_b5 ^
  --layers blocks.2 blocks.4
```

`--preprocess none` matches the existing legacy runtime assumption, but is only
correct when training also used unnormalized RGB `[0,1]`. The converter rejects
an empty PatchCore memory bank and strictly loads state dictionaries.

### PatchCore `.onnx` to `.engine`

```bat
python convert_to_tensorrt.py patchcore ^
  --input D:\models\patchcore\model_runtime.onnx ^
  --output models\patchcore\model_fp16.engine ^
  --preprocess none ^
  --input-size 256 256 ^
  --dynamic --min-batch 1 --batch 8 --max-batch 8
```

An existing PatchCore ONNX is accepted only as a canonical runtime graph that
already consumes NCHW RGB `[0,1]`, so it requires `--preprocess none`. The tool
does not pretend to add ImageNet/custom normalization to an arbitrary graph;
use the `.pt` route to embed those operations. A static batch-1 PatchCore ONNX
is rejected for the current batch-8 API. Re-export a dynamic ONNX from `.pt`
instead of editing only the ONNX input dimension.

Every successful conversion writes `<model>.engine.json` with source/engine
SHA256, exact profile and bindings, preprocessing contract, TensorRT/CUDA/GPU
versions, and the requested/actual precision. Existing outputs are protected;
use `--force` for an intentional atomic replacement.

On Windows, `04_convert_to_tensorrt.bat` resolves the same project Python 3.12
runtime as the service launchers and forwards all CLI arguments, for example:

```bat
04_convert_to_tensorrt.bat yolo --input pad.pt --output pad.engine --task detect
```

## How to Run

```bash
# Durable pipeline: run each command in a separate terminal/process
uv run python -m app.pipeline ingest
uv run python -m app.pipeline inference
uv run python -m app.pipeline publisher

# Process at most one claim and exit (diagnostics/tests)
uv run python -m app.pipeline ingest --once
uv run python -m app.pipeline inference --once --worker-id manual-inference
uv run python -m app.pipeline publisher --once

# Manually submit a historical date through the same durable pipeline
uv run python -m app.pipeline ingest --once --date 2026-08-26

# Explicit deployment config (AI_CONFIG_PATH remains supported)
uv run python -m app.pipeline publisher --config D:/path/to/ai_server.json

# Optional compatibility API; binds server.host:port (default 127.0.0.1:5050)
uv run python -m app.main

# Model servers + optional compatibility API; does not start scan_jobs.py
02_api_services.bat
```

Windows convenience launchers for the three independent stages are:

```bat
03_pipeline_ingest.bat
03_pipeline_inference.bat
03_pipeline_publisher.bat
```

All Windows launchers call `resolve_python.bat`: an explicitly configured
`PYTHON_EXE` wins, followed by `.venv`, the setup-created Conda environment, and
finally Python on `PATH`. Python 3.12 is validated before startup. The launchers
also default `AI_CONFIG_PATH` to the checked-in config. `02_api_services.bat` no
longer starts the legacy scanner; `02_scan.bat` requires the explicit safety flag
`SPI_ENABLE_LEGACY_SCANNER=1` and must be used only while durable Ingest is stopped.

Start Publisher first, then required model services (`8000` and `8002`), Inference,
and finally Ingest. Publisher must stay available even when models are down because
it owns the deadline fallback.

Each stage writes a separate system log derived from `logging.system_log_file`, for
example `system.ingest`, `system.inference`, and `system.publisher` under
`logging.log_dir`. The legacy entry point `uv run python ai_server_fastapi.py`
remains available but is deprecated.

## Deployment / Operations

### Windows services with NSSM

Use one service per worker so failure or restart of one stage does not stop the
others. Example, from an elevated command prompt:

```bat
set REPO=D:\Dre\JQ_SPI_02_AI_API
set PYTHON_EXE=%REPO%\.venv\Scripts\python.exe
set CONFIG=D:\Dre\JQ_SPI_02_AI_API\config\ai_server.json

nssm install SPI_Model_Anomaly "%PYTHON_EXE%" patchcore_api_trt.py --host 127.0.0.1 --port 8000
nssm set SPI_Model_Anomaly AppDirectory "%REPO%"
nssm set SPI_Model_Anomaly Start SERVICE_AUTO_START

nssm install SPI_Model_Distance "%PYTHON_EXE%" distance_detection_api_trt.py --host 127.0.0.1 --port 8002
nssm set SPI_Model_Distance AppDirectory "%REPO%"
nssm set SPI_Model_Distance Start SERVICE_AUTO_START

nssm install SPI_Pipeline_Ingest "%PYTHON_EXE%" -m app.pipeline ingest
nssm set SPI_Pipeline_Ingest AppDirectory "%REPO%"
nssm set SPI_Pipeline_Ingest AppEnvironmentExtra "AI_CONFIG_PATH=%CONFIG%"
nssm set SPI_Pipeline_Ingest Start SERVICE_AUTO_START

nssm install SPI_Pipeline_Inference "%PYTHON_EXE%" -m app.pipeline inference
nssm set SPI_Pipeline_Inference AppDirectory "%REPO%"
nssm set SPI_Pipeline_Inference AppEnvironmentExtra "AI_CONFIG_PATH=%CONFIG%"
nssm set SPI_Pipeline_Inference Start SERVICE_AUTO_START

nssm install SPI_Pipeline_Publisher "%PYTHON_EXE%" -m app.pipeline publisher
nssm set SPI_Pipeline_Publisher AppDirectory "%REPO%"
nssm set SPI_Pipeline_Publisher AppEnvironmentExtra "AI_CONFIG_PATH=%CONFIG%"
nssm set SPI_Pipeline_Publisher Start SERVICE_AUTO_START

nssm start SPI_Pipeline_Publisher
nssm start SPI_Model_Anomaly
nssm start SPI_Model_Distance
nssm start SPI_Pipeline_Inference
nssm start SPI_Pipeline_Ingest
```

If Python comes from Conda, set `PYTHON_EXE` to that environment's `python.exe`.
Before starting Inference/Ingest for the first production run, confirm both model
`/health` payloads report `status=healthy`; HTTP 200 alone is not sufficient while
a TensorRT engine is still initializing.
Configure NSSM stdout/stderr rotation as required by the plant's operations policy;
application event logs are already written under `logging.log_dir`.

Operational requirements and recovery behavior:

- **Local durable state**: keep `pipeline.database_path`,
  `paths.backup_output_root`, optional `pipeline.staging_root`, and
  `pipeline.result_root` on the AIPC's local disk. SQLite WAL is not supported for
  this design on the machine SMB share.
- **Crash recovery**: claims use `BEGIN IMMEDIATE`, owner IDs, attempt counters,
  and expiring leases. A restarted worker reclaims the earliest-deadline eligible
  job. Publisher uses a short heartbeat-renewed lease so a crash can be recovered
  inside the publish reserve. Atomic directories/manifests are reused when
  byte-identical.
- **At-least-once publication**: Primary CSV writes are atomic and deterministic,
  but there is no multi-file transaction. A crash halfway through a multi-CSV job
  can cause the retry to overwrite already returned CSVs with the same decisions.
- **Primary before backup**: `PRIMARY_RETURNED` is persisted immediately after the
  last Primary write. Local returned/processed backups run as a separate finalize
  claim; failure there never reruns inference.
- **Original before Primary**: Publisher never uses the live machine share as its
  input. The complete local raw copy must be available first, even if unusually
  slow local archival causes a soft-deadline miss.
- **Source immutability**: a timestamp `job_id` is the SQLite primary key. Do not
  modify or reuse a timestamp folder after it has passed readiness validation.
- **API probes**: `/health` and `/ready` belong to the compatibility API and do not
  currently report worker heartbeats or SQLite queue state.
- **Graceful shutdown**: stop between claims when practical. Publisher failure is
  recovered after `publisher_lease_seconds`; other stages use
  `worker_lease_seconds` or the deadline cutoff.
- **Compatibility metrics log rotation**: `log/log.csv` rotates at
  `logging.request_log_max_bytes` (default 50 MB) keeping
  `logging.request_log_backup_count` backups (`log.csv.1..N`); set
  `request_log_max_bytes: 0` to disable. No data is lost until the backup count is
  exceeded.
- **Config per environment**: keep deployment paths out of the shared repo by
  pointing `AI_CONFIG_PATH` at a machine-local config (template:
  `config/ai_server.example.json`).
- **Legacy scanner retry / dead-letter**: `scan_jobs.py` no longer re-posts a failing job
  on every tick. Client errors (4xx, e.g. a malformed CSV) are dead-lettered after
  `scanner_client_error_max_attempts` (default 2); server/transport errors (5xx or
  connection failures) get exponential backoff (`scanner_backoff_base_seconds` ..
  `scanner_backoff_max_seconds`) up to `scanner_max_retries` (default 5), then
  dead-letter. Dead-letter state is in-memory, so a scanner restart re-attempts.
- **CI**: `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, and
  the `tests/unit` + `tests/integration` suites on every push/PR.

## How to Test

```bash
uv run pytest
```

Unit tests (`tests/unit/`) cover ingest validation/atomic archive, SQLite claims and
cutoff fencing, result manifests, inference fallback, Publisher ordering, domain
services, and config. Integration tests drive both the compatibility `/process`
route and all three durable worker stages with temporary local/share paths and fake
model runners — no production paths or real endpoints are required.

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

- **Worker exits immediately**: set `pipeline.enabled=true` and inspect its startup
  preflight error. Ingest requires a readable `pipeline.watch_root`; every stage
  probes its required local/output directories.
- **No jobs reach `READY`**: inspect `system.ingest` for missing/empty/undecodable
  expected images, duplicate image keys, CSV template errors, or a source snapshot
  that keeps changing. The worker intentionally treats these as not-ready and
  scans again.
- **SQLite is locked or unreliable**: confirm `pipeline.database_path` is a local
  AIPC file, not a UNC/SMB path, and that all workers use the same database.
- **Job remains `PUBLISHING` or `PRIMARY_RETURNED`**: inspect `system.publisher`.
  `PUBLISHING` normally indicates Primary/SMB retry; `PRIMARY_RETURNED` means the
  machine result is already visible and only local `ai_result` finalization remains.
- **Unexpected all-23 result**: inspect the backed-up `ai_result/manifest.json`.
  `reason=publish_reserve_reached` means no durable inference result was committed
  before the cutoff; `required_model_failure`, `required_model_timeout`, and
  `images_exceed_threshold` identify the other fail-safe paths.
- **Required-model fallback**: inspect `errors` and `reason`; missing or invalid
  keys for the SINIC image-name template intentionally return all rows as `23`.
- **`400 CSV missing required columns`**: check the configured image-name
  template fields, normally `component_name`, `Array_id`, and `Pad_id`.
- **`high cover` never fires**: `cover%` needs `paste_pixels` (paste model) and
  `pad_area` (needs `Width` + `Length`). Paste is disabled by default.
- **Primary write fails**: check `external_output_root` and SMB availability. The
  Publisher retains a recoverable `PUBLISHING` state and retries; the 30-second SLA
  cannot be guaranteed during a network outage.
- **Compatibility `/process` returns `500`**: check `external_output_root` and
  `backup_output_root`; synchronous output failures return an HTTP error.
- **Config errors on startup**: `config/ai_server.json` must be present and valid
  (or set `AI_CONFIG_PATH`).

## Known Behavior and Compatibility Notes

- **Paste model (8001) is disabled by default.** It is a heavier YOLO + MobileSAM
  pipeline and the `cover%` / `high cover` rule depends on it. To enable: install
  the `paste` extra, start the paste server on `8001`, set the `paste` client's
  `"enabled": true`, and restart the merge server.
- **`short distance` uses `min_pad_distance < short_distance_threshold`** (less
  than), not greater than.
- **Compatibility `primary_csv_mode`** has two modes: `is_pass_only` (default) and
  `full_ai_columns` (see Output Files). The durable Publisher always uses the
  machine-return contract.
- **`*_processed.csv`** contains the full AI schema and exists for **debugging**.
- **Image count over `folder_images_num_threshold`**: inference is skipped and every
  row is marked `is_pass=23`. The durable path records
  `reason=images_exceed_threshold` in its manifest; `/process` retains the legacy
  `{"status": "finished scanning", "skipped": true, ...}` response.
- **Required model failure policy**: an endpoint error, deadline timeout,
  incomplete key set, or invalid scalar produces a fail-safe result with every row
  set to `23`. Optional-model failures continue normally. On the durable path this
  policy is recorded in the result manifest; on `/process` it is an HTTP-200 body.
- **Durable and synchronous paths are separate**: `/process` does not enqueue into
  SQLite, and the pipeline workers do not require the FastAPI server. Do not point
  both `scan_jobs.py` and Ingest Worker at the same production share.
- **Automatic Ingest scans only today's timestamp folders.** Historical and
  cross-midnight recovery remains manual; use
  `app.pipeline ingest --once --date YYYY-MM-DD`.
- **No queue can guarantee normal AI output under overload.** Earliest-deadline
  scheduling and cutoff takeover guarantee the configured all-23 decision path,
  subject to AIPC and SMB availability.
- **The SQLite database is job state, not an audit-event ledger.** Stage timestamps,
  attempts, reasons, and leases are persisted; detailed evidence remains in the
  stage-specific text logs and result manifest.
- **`log/`, `backup/`, `data/`** and model weights are git-ignored.
- **The compatibility API binds `127.0.0.1` by default** (`server.host`). Set it to
  a wider interface only if a remote API client is intentionally allowed.
- **Compatibility `/process` offloads CPU/IO-bound work** (CSV read/write and
  classification) to a thread pool, so `/health` and `/ready` stay responsive while
  a job runs.
- **Output CSVs are written atomically** (temp file + `os.replace`), so a crash
  mid-write never leaves a partial CSV at the target path.
- **Legacy code**: `ai_server_fastapi.py` (and the model servers / scanner) remain
  for the legacy entry point and are excluded from Ruff/mypy. The new `app/` entry
  point has no runtime dependency on `ai_server_fastapi.py`.
