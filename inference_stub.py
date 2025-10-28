import hashlib
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel


app = FastAPI(title="Inference Stub", version="1.0.0")


class JobRequest(BaseModel):
    job_folder: str


def build_img_name_column(df: pd.DataFrame) -> pd.Series:
    if "Array_id" not in df.columns or "Pad_no" not in df.columns:
        raise ValueError("CSV missing required columns: Array_id, Pad_no")
    array_minus_one = pd.to_numeric(df["Array_id"], errors="coerce").astype("Int64") - 1
    pad_str = df["Pad_no"].astype(str)
    return array_minus_one.astype(str) + "_" + pad_str + ".jpg"


def _value_for_img(img: str, port: int) -> float:
    # Deterministic value based on name and target port
    h = hashlib.md5(img.encode("utf-8")).hexdigest()
    base = int(h[:8], 16) / 0xFFFFFFFF  # 0..1
    if port == 8001:
        return round(base * 1000.0, 3)  # paste_pixels-like scale
    if port == 8002:
        return round(base * 10.0, 6)    # min_pad_distance-like scale
    return round(base, 6)               # anomaly_score-like scale


@app.post("/inference")
def inference(req: JobRequest, request: Request) -> Dict[str, Any]:
    folder = Path(req.job_folder)
    csv_files: List[Path] = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not csv_files:
        return {"results": {}}  # empty results for robustness

    # Use the first CSV to synthesize a result per img
    df = pd.read_csv(csv_files[0])
    if "img_name" not in df.columns:
        df = df.copy()
        df["img_name"] = build_img_name_column(df)

    # Simple deterministic scalar results per service port
    port = request.url.port or 0
    results = {img: _value_for_img(img, port) for img in df["img_name"].astype(str).tolist()}
    return {"results": results}
