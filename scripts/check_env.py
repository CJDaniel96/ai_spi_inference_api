#!/usr/bin/env python3
"""Environment verification for the SPI AI inference API.

Prints the interpreter / OS, then checks that the key runtime dependencies can
be imported and reports PyTorch / CUDA availability.

Usage:
    uv run python scripts/check_env.py

Behaviour notes:
    * macOS is treated as a CPU / Apple-Silicon profile: CUDA is NOT required and
      its absence is reported as informational, not an error.
    * Windows / Linux are treated as the CUDA deployment profile: if CUDA is not
      available a clear WARNING is printed, but the script does NOT crash.
    * The process exits non-zero only when a *required base* package is missing,
      so it is safe to use as a CI / smoke-test gate.
"""

from __future__ import annotations

import importlib
import platform
import sys

# ANSI colours (fall back to plain text when stdout is not a TTY).
_TTY = sys.stdout.isatty()
GREEN = "\033[32m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
BOLD = "\033[1m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""

OK = f"{GREEN}[ OK ]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{GREEN}[INFO]{RESET}"


def _header(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")


def check_import(module: str, *, required: bool, label: str | None = None) -> bool:
    """Try to import *module*; print an aligned status line. Returns success."""
    name = label or module
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
        status = FAIL if required else WARN
        kind = "required" if required else "optional"
        print(f"  {status} import {name:<16} ({kind}) -> {type(exc).__name__}: {exc}")
        return False
    version = getattr(mod, "__version__", "unknown")
    print(f"  {OK} import {name:<16} version={version}")
    return True


def main() -> int:
    system = platform.system()  # 'Darwin' | 'Windows' | 'Linux'
    is_macos = system == "Darwin"
    # Windows / Linux are the CUDA deployment profile.
    cuda_profile = system in ("Windows", "Linux")

    if is_macos:
        profile = "macOS / CPU"
    elif cuda_profile:
        profile = "CUDA (Windows/Linux)"
    else:
        profile = system

    _header("Interpreter / OS")
    print(f"  {INFO} Python  : {platform.python_version()} ({sys.executable})")
    print(f"  {INFO} OS      : {platform.platform()}")
    print(f"  {INFO} Machine : {platform.machine()}")
    print(f"  {INFO} Profile : {profile}")

    required_ok = True

    _header("Core dependencies")
    required_ok &= check_import("fastapi", required=True)
    required_ok &= check_import("uvicorn", required=True)
    required_ok &= check_import("pydantic", required=True)
    required_ok &= check_import("httpx", required=True)

    _header("Data processing")
    required_ok &= check_import("pandas", required=True)
    required_ok &= check_import("numpy", required=True)

    _header("Computer vision")
    required_ok &= check_import("cv2", required=True, label="cv2 (opencv)")
    check_import("matplotlib", required=False)

    _header("AI / ML")
    torch_ok = check_import("torch", required=True)
    required_ok &= torch_ok
    check_import("ultralytics", required=False)

    # ONNX Runtime: GPU build on CUDA hosts, CPU build on macOS. Either import
    # exposes the same top-level module name, so a single check covers both.
    check_import("onnxruntime", required=False)

    # CUDA report (only meaningful once torch imported).
    _header("PyTorch / CUDA")
    if torch_ok:
        import torch  # already imported above; cheap re-import

        print(f"  {INFO} torch version   : {torch.__version__}")
        cuda_available = bool(torch.cuda.is_available())
        print(f"  {INFO} torch.cuda.is_available() : {cuda_available}")

        if cuda_available:
            print(f"  {OK} CUDA version    : {torch.version.cuda}")
            try:
                count = torch.cuda.device_count()
                for i in range(count):
                    print(f"  {OK} GPU[{i}]          : {torch.cuda.get_device_name(i)}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {WARN} could not query GPU name: {exc}")
        else:
            # Apple Silicon MPS is a nice-to-know on macOS.
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if is_macos and mps is not None and mps.is_available():
                print(f"  {INFO} Apple MPS (Metal) backend is available.")
            if is_macos:
                print(
                    f"  {INFO} CUDA is not available on this machine. This is "
                    "expected for macOS / CPU-only development."
                )
            elif cuda_profile:
                print(
                    f"  {WARN} CUDA is not available on this machine. This is "
                    "acceptable for macOS or CPU-only development, but "
                    "Windows/Linux CUDA deployment should verify GPU separately."
                )
            else:
                print(f"  {INFO} CUDA is not available (unrecognised platform).")
    else:
        print(f"  {FAIL} torch is not importable; skipping CUDA checks.")

    # Summary / exit code.
    _header("Summary")
    if required_ok:
        print(f"  {OK} All required base packages import successfully.")
        code = 0
    else:
        print(
            f"  {FAIL} One or more REQUIRED packages failed to import. "
            "Check that the environment was synced for this platform."
        )
        code = 1
    print()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
