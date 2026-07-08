"""Application entry point (experimental new architecture).

Creates the FastAPI application, wires the health and process routes, and
configures logging. This is the new modular-monolith entry point and is meant
to run as a drop-in for the legacy ``ai_server_fastapi.py`` server (same
``/health`` and ``/process`` contract on port 5050) during migration.

Run with::

    python -m app.main
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health as health_route
from app.api.routes import process as process_route
from app.core.errors import AppError
from app.core.logging import setup_logging

_HOST = "0.0.0.0"
_PORT = 5050


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    setup_logging()

    application = FastAPI(title="SPI AI Inference API", version="0.1.0")
    application.include_router(health_route.router)
    application.include_router(process_route.router)

    @application.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Convert application errors into JSON responses."""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_HOST, port=_PORT)
