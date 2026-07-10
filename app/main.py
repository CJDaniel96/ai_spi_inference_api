"""Application entry point (modular, layered architecture).

Creates the FastAPI application, wires the health and process routes, configures
logging from config, and registers centralized exception handlers. This is the
primary entry point (``python -m app.main``, port 5050); the legacy
``ai_server_fastapi.py`` server is kept only for backward compatibility.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health as health_route
from app.api.routes import process as process_route
from app.core.config import get_config, resolve_under_project_root
from app.core.errors import AppError
from app.core.logging import get_logger, setup_logging

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5050
_INTERNAL_ERROR_DETAIL = "Internal Server Error"
_SERVER_ERROR_STATUS = 500


def _server_bind() -> tuple[str, int]:
    """Return the (host, port) to bind, from config with a safe fallback."""
    try:
        server = get_config().server
        return server.host, server.port
    except Exception:  # noqa: BLE001 - fall back to loopback defaults
        return _DEFAULT_HOST, _DEFAULT_PORT


def _configure_logging() -> None:
    """Configure logging from config, falling back to defaults on any failure."""
    try:
        config = get_config()
        setup_logging(
            log_dir=resolve_under_project_root(config.logging.log_dir),
            system_log_file=config.logging.system_log_file,
        )
    except Exception:  # noqa: BLE001 - never let logging setup block startup
        setup_logging()


def _register_exception_handlers(application: FastAPI) -> None:
    """Map typed app errors to responses and log unexpected errors."""
    logger = get_logger("api")

    @application.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """Return a typed error's ``detail``; log server-side (5xx) failures."""
        if exc.status_code >= _SERVER_ERROR_STATUS:
            logger.error(
                "event=api.error path=%s status=%d err=%s",
                request.url.path,
                exc.status_code,
                exc.message,
            )
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.message}
        )

    @application.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Log the traceback and return a generic 500 without leaking details."""
        logger.exception(
            "event=api.unhandled path=%s method=%s err=%s",
            request.url.path,
            request.method,
            str(exc),
        )
        return JSONResponse(
            status_code=_SERVER_ERROR_STATUS,
            content={"detail": _INTERNAL_ERROR_DETAIL},
        )


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Own a shared HTTP client for the app's lifetime (graceful shutdown)."""
    application.state.http_client = httpx.AsyncClient()
    try:
        yield
    finally:
        await application.state.http_client.aclose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    _configure_logging()
    application = FastAPI(
        title="AI SPI Inference API", version="0.1.0", lifespan=_lifespan
    )
    application.include_router(health_route.router)
    application.include_router(process_route.router)
    _register_exception_handlers(application)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host, port = _server_bind()
    uvicorn.run(app, host=host, port=port)
