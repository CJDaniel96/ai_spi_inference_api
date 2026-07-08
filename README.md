# Four-Service AI Merge Pipeline (FastAPI)

This repository provides four FastAPI services that work together to process a job folder, call three AI endpoints in parallel, merge their numeric results into the CSV(s), classify defects, and save a merged CSV into an `AI` subfolder.

- Merge server: `ai_server_fastapi.py` on port `5050` (`POST /process`)
- Model servers:
  - `8000` PatchCore anomaly — returns `anomaly_score`
  - `8001` Paste detection — returns `paste_pixels`
  - `8002` Distance detection — returns `min_center_to_pad_distance`

## Environment Management with uv

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python
environment and dependencies. `pyproject.toml` is the **canonical** dependency
source; the `requirements-*.txt` files are pinned mirrors kept for legacy /
non-uv deployment (see below).

Three platforms are supported:

| Platform          | PyTorch build                | Extras installed        |
| ----------------- | ---------------------------- | ----------------------- |
| Windows + CUDA    | CUDA wheels (PyTorch index)  | `cuda` (+ `dev`)        |
| Linux + CUDA      | CUDA wheels (PyTorch index)  | `cuda` (+ `dev`)        |
| macOS (CPU / MPS) | PyPI wheels (CPU / Apple MPS)| `mac` (+ `dev`)         |

> **Python version:** `>=3.10,<3.12` (3.10 or 3.11). Some pinned dependencies do
> not yet support 3.12, so do not use 3.12.

### Prerequisite: install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv --version
```

### Dependency organisation

- **base** (`[project].dependencies`): cross-platform packages (fastapi,
  uvicorn, httpx, pydantic, pandas, numpy, opencv-python, matplotlib,
  ultralytics, anomalib, torch, torchvision).
- **`cuda` extra**: Windows/Linux CUDA-only packages — `onnxruntime-gpu`,
  `pycuda` (markered so they are never resolved/installed on macOS).
  TensorRT is installed out-of-band; see the CUDA notes below.
- **`mac` extra**: `onnxruntime` (CPU build) — no CUDA-only packages.
- **`dev` group**: `pytest`, `ruff`, `mypy`.
- **`analytics` extra** (optional): `plotly`, `xlsxwriter` for the analytics
  scripts under `test/`.
- **`paste` extra** (optional): MobileSAM (git dependency) for the paste
  detection server (port 8001).

### Windows + CUDA

```powershell
uv venv --python 3.10
.venv\Scripts\activate

# 1) Install the CUDA build of PyTorch from the PyTorch index (do this FIRST):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2a) Reproducible / legacy path (recommended for deployment):
uv pip install -r requirements-cuda.txt
uv pip install -r requirements-dev.txt

# 2b) …or the pyproject-driven path:
uv sync --extra cuda --group dev
```

> `uv sync` installs the default PyPI build of torch, which on **Windows is
> CPU-only**. On Windows always run step (1) (the PyTorch CUDA index) so the
> CUDA wheels are in place; `uv sync` will then keep them.

### Linux + CUDA

```bash
uv venv --python 3.10
source .venv/bin/activate

# On Linux the default PyPI torch already bundles CUDA. To pin a specific
# CUDA version, install from the PyTorch index first:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Reproducible / legacy path:
uv pip install -r requirements-cuda.txt
uv pip install -r requirements-dev.txt

# …or the pyproject-driven path:
uv sync --extra cuda --group dev
```

### macOS (CPU / Apple Silicon)

macOS does **not** install any CUDA-only dependency (`pycuda`,
`onnxruntime-gpu`, `tensorrt`).

```bash
uv venv --python 3.10
source .venv/bin/activate

# pyproject-driven path (recommended):
uv sync --extra mac --group dev

# …or the reproducible / legacy path:
uv pip install -r requirements-mac.txt
uv pip install -r requirements-dev.txt
```

torch / torchvision come from PyPI (CPU + Apple Silicon MPS) — no index needed.

### CUDA / PyTorch notes

- **PyTorch index URL** is only needed for CUDA wheels on Windows/Linux; it is
  deliberately kept out of the universal dependency set. Pick the CUDA build
  that matches your driver, e.g. `cu121` or `cu124`.
- **TensorRT** is imported by `patchcore_inf_trt.py` but is installed
  out-of-band from NVIDIA to match your CUDA toolkit — pin it in
  `requirements-cuda.txt` for your deployment (see the TODO in that file).
- **Legacy discrepancy (TODO):** `setup.bat` installs `torch 1.13.1+cu117`
  while `requirements.txt` comments target `cu121` and `onnxruntime-gpu==1.19.0`
  needs CUDA 12.x. Standardise on one CUDA toolchain before production rollout.

### Common uv commands

```bash
uv sync                              # install base deps into .venv
uv sync --group dev                  # base + dev tooling
uv sync --extra cuda --group dev     # Windows/Linux CUDA + dev
uv sync --extra mac --group dev      # macOS + dev

uv run python ai_server_fastapi.py   # legacy entry point (merge server, 5050)
# uv run python -m app.main          # future entry point (after refactor)

uv run pytest
uv run ruff check .
uv run ruff format .
uv run python scripts/check_env.py   # verify the environment
```

### requirements-*.txt (legacy / deployment)

`pyproject.toml` is the source of truth. These pinned files mirror it for
non-uv `pip` installs and reproducible CI/deployment images:

- `requirements-cuda.txt` — base + CUDA-only packages (Windows/Linux)
- `requirements-mac.txt` — base + CPU onnxruntime (macOS)
- `requirements-dev.txt` — dev tooling (pytest / ruff / mypy)
- `requirements.txt` — original legacy conda list (kept as-is)

Keep them in sync with `pyproject.toml`; do not let them diverge long-term.

### uv.lock

`uv.lock` is currently **git-ignored**. A single universal lockfile cannot
faithfully capture the CUDA build of torch or the TensorRT bindings (both
installed out-of-band), so reproducible installs use the pinned
`requirements-*.txt` files instead. Revisit committing `uv.lock` once the team
standardises on one CUDA toolchain and brings torch fully under uv.

## How To Use The API

- Start services (Windows): run `start_fastapi_services.bat`.
  - It launches four consoles on ports 5050, 8000, 8001, 8002.
  - Use forward slashes in JSON paths on Windows (e.g., `D:/path/...`).

- Merge server (5050)
  - Endpoint: `POST /process`
  - Body: `{ "job_folder": "D:/path/to/job" }`
  - bash:
    - `curl.exe -X POST "http://127.0.0.1:5050/process" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:/Dre/JQ_SPI_02_AI_API/data/20250523135357\" }"`
  - PowerShell:
    - `$body = @{ job_folder = 'D:/Dre/JQ_SPI_02_AI_API/data/20250523135357' } | ConvertTo-Json`
    - `Invoke-RestMethod -Uri 'http://127.0.0.1:5050/process' -Method Post -ContentType 'application/json' -Body $body`

- PatchCore anomaly (8000)
  - Endpoint: `POST /inference`
  - Body: `{ "job_folder": "D:/path/to/job" }`
  - Health: `GET /health`

- Paste detection (8001)
  - Endpoint: `POST /inference`
  - Body: `{ "job_folder": "D:/path/to/job" }`
  - Health: `GET /health`

- Distance detection (8002)
  - Endpoint: `POST /inference`
  - Body: `{ "job_folder": "D:/path/to/job" }`
  - Health: `GET /health`

## Service Returns

- Merge server (5050 `POST /process`)
  - Returns: `{ "status": "ok", "saved_files": [".../AI/xxx.csv", ...], "errors": ["..."], "csv_count": <int> }`
  - Side effects: writes merged CSV(s) to `AI/` subfolder of the job.

- PatchCore anomaly (8000 `POST /inference`)
  - Returns: `{ "status": "success", "message": str, "total_images": int, "anomalies_detected": int, "average_score": float, "processing_time": float, "results": { "<img>.jpg": <float|null>, ... } }`
  - Key `results` maps filename to `anomaly_score` (NaN -> null).

- Paste detection (8001 `POST /inference`)
  - Returns: `{ "status": "success", "message": str, "total_images": int, "total_inference_time": float, "results": { "<stem>.jpg": <float|null>, ... } }`
  - Keys are normalized to `<stem>.jpg` to match merge logic; values are `paste_pixels` (NaN -> null).

- Distance detection (8002 `POST /inference`)
  - Returns: `{ "status": "success", "message": str, "total_images": int, "total_inference_time": float, "results": { "<img>.jpg": <float|null>, ... } }`
  - Values are `min_center_to_pad_distance` (NaN -> null).

## Quick Start

1) Activate env and install deps
- `conda activate spi_env`
- `pip install -r requirements_paste_detection.txt`

2) Start all services (Windows)
- Run `start_fastapi_services.bat` (opens four consoles)

3) Run a job
- Call the merge server as shown in the API section above.

## Output Columns (from merge)

- `img_name` from `Array_id` and `Pad_no` → `{Array_id-1}_{Pad_no}.jpg`
- `anomaly_score` from port 8000
- `paste_pixels` from port 8001
- `min_pad_distance` from port 8002
- `pad_area` = `pi * Width * Length / 4` (when `Width`, `Length` exist)
- `cover%` = `paste_pixels * 0.8246 * 100 / pad_area` (when `pad_area` exists)
- `ai_defect_name` per ordered rules (below)
- `is_pass` = `22` if `ai_defect_name` empty else `23`

## Defect Rules (order matters)

- Anomaly: `anomaly_score > 0.5` → `FM/color`
- Volume: `insp_vol > vol_h_ng + 20` → `high vol`; `insp_vol < vol_l_ng - 10` → `low vol`
- Coverage: `cover% > 180` → `high cover`
- Distance: `min_pad_distance > 6.6` → `short distance`
- Paste height: `insp_height > 200` → `high paste`

## Troubleshooting

- Empty columns: ensure model responses use the same `img_name` format.
- Missing derived columns: `pad_area` needs `Width` + `Length`; `cover%` needs `pad_area`.
- `is_pass` depends on whether `ai_defect_name` is empty after rules.

