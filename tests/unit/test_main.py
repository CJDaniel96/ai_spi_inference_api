"""Unit tests for app.main bind resolution."""

from __future__ import annotations

from app.core.config import get_config
from app.main import _DEFAULT_HOST, _DEFAULT_PORT, _server_bind


def test_server_bind_defaults_to_loopback() -> None:
    assert _DEFAULT_HOST == "127.0.0.1"
    assert _DEFAULT_PORT == 5050


def test_server_bind_reads_config_host_port() -> None:
    get_config.cache_clear()
    host, port = _server_bind()
    assert host == "127.0.0.1"
    assert port == 5050
