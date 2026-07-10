"""Unit tests for app.main bind resolution."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_config
from app.main import _DEFAULT_HOST, _DEFAULT_PORT, _server_bind, create_app


def test_server_bind_defaults_to_loopback() -> None:
    assert _DEFAULT_HOST == "127.0.0.1"
    assert _DEFAULT_PORT == 5050


def test_server_bind_reads_config_host_port() -> None:
    get_config.cache_clear()
    host, port = _server_bind()
    assert host == "127.0.0.1"
    assert port == 5050


def test_lifespan_opens_and_closes_shared_http_client() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        shared = app.state.http_client
        assert shared.is_closed is False
    assert shared.is_closed is True
