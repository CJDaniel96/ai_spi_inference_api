import asyncio
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="AI Merge Server", version="1.0.0")


class JobRequest(BaseModel):
    job_folder: str


def build_img_name_column(df: pd.DataFrame) -> pd.Series:
    if "Array_id" not in df.columns or "Pad_no" not in df.columns:
        missing = [c for c in ("Array_id", "Pad_no") if c not in df.columns]
        raise ValueError(f"CSV missing required columns: {missing}")
    array_minus_one = pd.to_numeric(df["Array_id"], errors="coerce").astype("Int64") - 1
    pad_str = df["Pad_no"].astype(str)
    return array_minus_one.astype(str) + "_" + pad_str + ".jpg"


async def post_job(client: httpx.AsyncClient, url: str, job_folder: str, timeout: int = 300) -> Tuple[str, Dict[str, Any], str]:
    try:
        resp = await client.post(url, json={"job_folder": job_folder}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", {})
        if not isinstance(results, dict):
            return url, {}, f"Unexpected results type from {url}: {type(results)}"
        return url, results, ""
    except Exception as exc:  # noqa: BLE001
        return url, {}, f"Request to {url} failed: {exc}"


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
    errors: List[str] = []

    async with httpx.AsyncClient() as client:
        tasks = [post_job(client, url, job_folder) for url, _ in endpoints]
        for url, res, err in await asyncio.gather(*tasks):
            if err:
                errors.append(err)
            results_by_endpoint[url] = res

    ai_dir = folder / "AI"
    ai_dir.mkdir(parents=True, exist_ok=True)

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
        df.to_csv(out_path, index=False)
        saved_files.append(str(out_path))

    return {"saved_files": saved_files, "errors": errors, "csv_count": len(csv_files)}


# ------------------------
# Post-merge enrich helpers
# ------------------------

ANOMALY_THRESHOLD = 0.5
HIGH_COVER_THRESHOLD = 180.0
SHORT_DISTANCE_THRESHOLD = 6.6
LOW_VOL_OFFSET = -10.0
HIGH_VOL_OFFSET = 20.0
HIGH_PASTE_HEIGHT_THRESHOLD = 200.0


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

    # 1) Anomaly threshold
    if "anomaly_score" in out.columns:
        anomaly = pd.to_numeric(out["anomaly_score"], errors="coerce")
        mask = (~assigned) & (anomaly > ANOMALY_THRESHOLD)
        out.loc[mask, "ai_defect_name"] = "FM/color"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 2) insp_vol thresholds using offsets and vol_l_ng, vol_h_ng
    cols_needed = {"insp_vol", "vol_l_ng", "vol_h_ng"}
    if cols_needed.issubset(out.columns):
        insp_vol = pd.to_numeric(out["insp_vol"], errors="coerce")
        vol_l_ng = pd.to_numeric(out["vol_l_ng"], errors="coerce")
        vol_h_ng = pd.to_numeric(out["vol_h_ng"], errors="coerce")
        low_thr = vol_l_ng + LOW_VOL_OFFSET
        high_thr = vol_h_ng + HIGH_VOL_OFFSET

        mask_high = (~assigned) & (insp_vol > high_thr)
        out.loc[mask_high, "ai_defect_name"] = "high vol"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

        mask_low = (~assigned) & (insp_vol < low_thr)
        out.loc[mask_low, "ai_defect_name"] = "low vol"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 3) high cover
    if "cover%" in out.columns:
        cover = pd.to_numeric(out["cover%"], errors="coerce")
        mask = (~assigned) & (cover > HIGH_COVER_THRESHOLD)
        out.loc[mask, "ai_defect_name"] = "high cover"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 4) short distance (note: condition provided as 6.6 < min_pad_distance)
    if "min_pad_distance" in out.columns:
        dist = pd.to_numeric(out["min_pad_distance"], errors="coerce")
        mask = (~assigned) & (dist > SHORT_DISTANCE_THRESHOLD)
        out.loc[mask, "ai_defect_name"] = "short distance"
        assigned = out["ai_defect_name"].astype(str).str.len() > 0

    # 5) high paste based on insp_height
    if "insp_height" in out.columns:
        height = pd.to_numeric(out["insp_height"], errors="coerce")
        mask = (~assigned) & (height > HIGH_PASTE_HEIGHT_THRESHOLD)
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
    try:
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
