import asyncio
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
        out_path = ai_dir / csv_path.name
        df.to_csv(out_path, index=False)
        saved_files.append(str(out_path))

    return {"saved_files": saved_files, "errors": errors, "csv_count": len(csv_files)}


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
