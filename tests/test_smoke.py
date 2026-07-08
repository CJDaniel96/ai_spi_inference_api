"""Smoke tests: verify the environment is wired up.

These do NOT exercise the inference pipeline — they only assert that the core
runtime dependencies import. Real unit tests can be added alongside them later.
"""

import importlib

import pytest

REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "httpx",
    "pandas",
    "numpy",
    "cv2",
    "torch",
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_required_module_imports(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_torch_reports_cuda_availability() -> None:
    """torch.cuda.is_available() must be callable (True or False, never raise)."""
    import torch

    assert isinstance(torch.cuda.is_available(), bool)
