# Four-Service AI Merge Pipeline (FastAPI)

This repository provides four FastAPI services that work together to process a job folder, call three AI endpoints in parallel, merge their numeric results into the CSV(s), classify defects, and save a merged CSV into an `AI` subfolder.

- Merge server: `ai_server_fastapi.py` on port `5050` (`POST /process`)
- Model servers (defaults provided as stubs):
  - `8000` anomaly-score → returns `anomaly_score` per `img_name`
  - `8001` paste-detection → returns `paste_pixels` per `img_name`
  - `8002` distance-detection → returns `min_pad_distance` per `img_name`

Each model endpoint must return JSON: `{ "results": { "<img_name>": <float>, ... } }`, where `img_name = {Array_id-1}_{Pad_no}.jpg`.

## Quick Start

1) Create/activate Conda env
- `conda activate trt`

2) Install dependencies
- `pip install -r requirements_paste_detection.txt`

3) Start all services (Windows)
- Easiest: run `start_fastapi_services.bat`
  - Activates `trt` and launches all 4 services in separate windows:
    - Merge: `http://127.0.0.1:5050/process`
    - Models: `http://127.0.0.1:8000/8001/8002`

Manual alternative (4 terminals after `conda activate trt`):
- `uvicorn ai_server_fastapi:app --host 0.0.0.0 --port 5050`
- `uvicorn inference_stub:app --host 0.0.0.0 --port 8000`
- `uvicorn inference_stub:app --host 0.0.0.0 --port 8001`
- `uvicorn inference_stub:app --host 0.0.0.0 --port 8002`

You can replace the three stub servers with your real apps (e.g., `paste_detection_server.py`, `distance_detection_server.py`) as long as they return the expected `results` mapping keyed by `img_name`.

## Run A Job

1) Prepare a folder with at least one CSV containing `Array_id` and `Pad_no` (used to construct `img_name`). Example:
- `D:/Dre/JQ_SPI_02_AI_API/data/20250523135357`

2) Call the merge server
- PowerShell:
  - `$body = @{ job_folder = 'D:/Dre/JQ_SPI_02_AI_API/data/20250523135357' } | ConvertTo-Json`
  - `Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:5050/process' -ContentType 'application/json' -Body $body`
- curl.exe:
  - `curl.exe -X POST "http://127.0.0.1:5050/process" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:/Dre/JQ_SPI_02_AI_API/data/20250523135357\" }"`

3) Output
- Merged CSV(s) saved to `.../AI/<same_name>.csv`
- `img_name` is added as `{Array_id-1}_{Pad_no}.jpg`

## Output Columns

The merge server writes or updates these columns when present/available:
- `img_name` → from `Array_id` and `Pad_no`
- `anomaly_score` → from port 8000
- `paste_pixels` → from port 8001
- `min_pad_distance` → from port 8002
- `pad_area` → `π × Width × Length / 4` (when `Width` and `Length` exist)
- `cover%` → `paste_pixels × 0.8246 × 100 / pad_area` (when `pad_area` exists)
- `ai_defect_name` → first matching rule below
- `is_pass` → `22` if `ai_defect_name` is empty, else `23`

## How ai_defect_name Is Determined (Order Matters)

The first matching condition per row wins, then the next row is evaluated:
- 1) Anomaly: if `anomaly_score > 0.5` → `FM/color`
- 2) Volume thresholds (requires `insp_vol`, `vol_l_ng`, `vol_h_ng`):
  - If `insp_vol > vol_h_ng + 20` → `high vol`
  - Else if `insp_vol < vol_l_ng - 10` → `low vol`
- 3) Coverage (requires `cover%`):
  - If `cover% > 180` → `high cover`
- 4) Distance (requires `min_pad_distance`):
  - If `min_pad_distance > 6.6` → `short distance`
- 5) Paste height (requires `insp_height`):
  - If `insp_height > 200` → `high paste`

Only the first satisfied rule is applied per row.

## Expected Model Responses

Each model endpoint must return scalar floats keyed by `img_name`:
- `8000` anomaly: `{ "results": { "<img_name>": <float> } }`
- `8001` paste: `{ "results": { "<img_name>": <float> } }`
- `8002` distance: `{ "results": { "<img_name>": <float> } }`

The included `inference_stub.py` produces this format for all three ports, so you can test end‑to‑end immediately.

## Troubleshooting

- Columns empty (NaN): verify model responses use the same `img_name` format (`{Array_id-1}_{Pad_no}.jpg`). A mismatch prevents merges.
- Missing derived columns: `pad_area` needs `Width` + `Length`; `cover%` needs `pad_area`; some rules require specific columns.
- is_pass not set as expected: it depends on whether `ai_defect_name` is empty after applying the rules.
