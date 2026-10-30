"""One HTTP boundary for resident center detection and area segmentation."""

from __future__ import annotations

import argparse
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.core.config import resolve_under_project_root
from app.infrastructure.deadline_io import DeadlineExceeded
from app.infrastructure.yolo_area import (
    ModelBusyError,
    YoloAreaEngine,
    YoloAreaSettings,
)

LOG = logging.getLogger("spi.area_service")


class AreaRequest(BaseModel):
    job_folder: str


def create_app(settings: YoloAreaSettings, *, engine=None) -> FastAPI:
    engine = engine or YoloAreaEngine(settings)
    startup_error = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal startup_error
        startup_error = None
        try:
            engine.load()
        except Exception as exc:
            startup_error = str(exc)
            LOG.exception("Area model startup failed")
        yield

    app = FastAPI(title="SPI Detection and Segmentation", lifespan=lifespan)

    @app.get("/health")
    def health():
        ready = engine.is_ready and startup_error is None
        return {
            "status": "healthy" if ready else "initializing",
            "model_ready": ready,
            "error": startup_error,
            "model_version": engine.model_version,
        }

    @app.post("/inference")
    def inference(
        request: AreaRequest, x_spi_deadline_at: float | None = Header(default=None)
    ):
        if startup_error or not engine.is_ready:
            raise HTTPException(503, detail=startup_error or "Models are not ready")
        if x_spi_deadline_at is not None and not math.isfinite(x_spi_deadline_at):
            raise HTTPException(422, detail="Deadline must be finite")
        folder = Path(request.job_folder)
        if not folder.is_dir():
            raise HTTPException(404, detail="Job folder not found")
        try:
            result = engine.process_folder(folder, deadline_at=x_spi_deadline_at)
            LOG.info(
                "job_id=%s result=%s",
                folder.name,
                json.dumps(result, ensure_ascii=False),
            )
            return result
        except ModelBusyError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        except DeadlineExceeded as exc:
            raise HTTPException(504, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/yolo_area.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    settings = YoloAreaSettings.model_validate_json(
        resolve_under_project_root(args.config).read_text(encoding="utf-8")
    )
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
