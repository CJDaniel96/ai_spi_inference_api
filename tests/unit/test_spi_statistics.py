"""Verify that SPI reports select dates from backup folders."""

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from app.tools.spi_statistics import main
from tests.helpers import make_config


@pytest.mark.parametrize(
    ("options", "days"),
    [
        (
            ["--start-date", "2026-11-13", "--end-date", "2026-11-14"],
            ("2026-11-13", "2026-11-14"),
        ),
        (["--start-date", "2026-11-14"], ("2026-11-14", "2026-11-15")),
        (["--end-date", "2026-11-13"], ("2026-11-12", "2026-11-13")),
        (["--date", "2026-11-13"], ("2026-11-13",)),
    ],
)
def test_report_range_uses_backup_folder_dates(tmp_path, options, days):
    config = make_config(tmp_path)
    config_file = tmp_path / "settings.json"
    config_file.write_text(config.model_dump_json(), encoding="utf-8")
    root = Path(config.paths.backup_output_root)
    first_seen = datetime.fromisoformat("2026-11-14T00:01:00+08:00").timestamp()
    for day in ("2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"):
        job_id = day.replace("-", "") + "235900"
        report = root / day / job_id / "ai_result" / "run.json"
        report.parent.mkdir(parents=True)
        run = {
            "run_id": day,
            "job_id": job_id,
            "first_seen_at": first_seen,
            "publications": {
                "board.csv": {
                    "status": "RETURNED",
                    "attempt_log": [
                        {"number": 1, "status": "RETURNED", "started_at": first_seen}
                    ],
                }
            },
        }
        report.write_text(json.dumps(run), encoding="utf-8")

    archived = root / "archive" / "job" / "ai_result" / "run.json"
    archived.parent.mkdir(parents=True)
    archived.write_text("invalid JSON in a non-date folder", encoding="utf-8")
    output = tmp_path / "reports"
    assert main(["--config", str(config_file), "--output", str(output), *options]) == 0
    for name in ("jobs.csv", "deliveries.csv"):
        with (output / name).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["run_id"] for row in rows] == list(days)
    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["run_count"] == str(len(days))


@pytest.mark.parametrize(
    "options",
    [
        ["--start-date", "2026-11-14", "--end-date", "2026-11-13"],
        ["--start-date", "2026-11-31"],
        ["--date", "2026-11-13", "--end-date", "2026-11-15"],
    ],
)
def test_invalid_report_date_options_exit_before_loading_config(options):
    with pytest.raises(SystemExit) as error:
        main(options)
    assert error.value.code == 2
