"""Single production entry point: python -m app.main."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

from app.core.config import load_config
from app.core.logging import get_logger, setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPI AI single-controller runtime")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Discover one batch and drain one bounded background attempt",
    )
    parser.add_argument(
        "--date", type=date.fromisoformat, help="Input date for --once, YYYY-MM-DD"
    )
    args = parser.parse_args(argv)
    if args.date and not args.once:
        parser.error("--date requires --once")
    try:
        config = load_config(args.config)
        if args.check_config:
            print("Configuration valid: single controller, native OBB, async audit DB")
            return 0
        setup_logging(
            Path(config.logging.log_dir), system_log_file=config.logging.system_log_file
        )
        from app.application.controller import Controller
        from app.infrastructure.inference.resident_process import ResidentObbRuntime

        runtime = ResidentObbRuntime(config.inference.obb)
        jobs = asyncio.run(
            Controller(config, runtime).run(once=args.once, target_date=args.date)
        )
        if args.once and any(
            job.status in {"EXPIRED", "FAILED", "PUBLICATION_UNKNOWN"}
            for job in jobs.values()
        ):
            return 2
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        get_logger().exception(
            "Controller stopped; local journal retained for recovery"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
