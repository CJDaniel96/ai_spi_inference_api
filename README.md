# Four-Service AI Merge Pipeline (FastAPI)

This repository provides four FastAPI services that work together to process a job folder, call three AI endpoints in parallel, merge their numeric results into the CSV(s), classify defects, and save a merged CSV into an `AI` subfolder.

- Merge server: `ai_server_fastapi.py` on port `5050` (`POST /process`)
- Model servers:
  - `8000` PatchCore anomaly — returns `anomaly_score`
  - `8001` Paste detection — returns `paste_pixels`
  - `8002` Distance detection — returns `min_center_to_pad_distance`

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
- `conda activate trt`
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

