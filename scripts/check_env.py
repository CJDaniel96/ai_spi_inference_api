"""Check the controller dependencies; GPU checks are explicitly requested."""

import argparse
import importlib
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(f"Python {platform.python_version()}: {sys.executable}")
    print(platform.platform())
    success = sys.version_info[:2] == (3, 12)
    names = ["pydantic", "pandas", "numpy", "cv2"]
    if args.inference or args.require_cuda:
        names += ["torch", "ultralytics"]
    loaded = {}
    for name in names:
        try:
            loaded[name] = importlib.import_module(name)
            print(f"OK {name}: {getattr(loaded[name], '__version__', 'available')}")
        except Exception as exc:
            print(f"ERROR {name}: {exc}")
            success = False
    if args.require_cuda:
        cuda = "torch" in loaded and loaded["torch"].cuda.is_available()
        print(f"CUDA available: {cuda}")
        success = success and cuda
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
