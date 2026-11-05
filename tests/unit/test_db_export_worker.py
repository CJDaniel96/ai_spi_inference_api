"""Background export must recover without holding up machine publication."""

import json

import pytest

from app.application.workers.db_export_worker import DbExportWorker
from app.infrastructure.deadline_io import DeadlineExceeded


class ControlledExportIO:
    def __init__(self):
        self.operation = None
        self.released = False
        self.error = None
        self.closed = False

    def submit(self, function, *args, deadline_at):
        assert self.operation is None
        self.operation = function, args

    def poll(self):
        if self.error is not None:
            raise self.error
        if not self.released:
            return False, None
        function, args = self.operation
        self.operation = None
        return True, function(*args)

    def close(self):
        self.operation = None
        self.closed = True


@pytest.fixture
def export_lane(tmp_path):
    backup = tmp_path / "backup"
    source = backup / "20261105120000_processed.csv"
    backup.mkdir()
    source.write_text("is_pass,ai_defect_name\n22,\n23,\n")
    destination = tmp_path / "upload"
    config_path = tmp_path / "export.json"
    config_path.write_text(
        json.dumps(
            dict(
                site="SX",
                factory="SX",
                line="SPI",
                station_id="SPI",
                input_path=str(tmp_path / "unused"),
                output_path=str(destination),
                source_settle_seconds=0,
                poll_seconds=2,
            )
        )
    )
    lane = ControlledExportIO()
    clock = [100.0]
    worker = DbExportWorker(
        config_path=config_path,
        backup_root=backup,
        log_dir=tmp_path / "logs",
        system_log_file="system.db_export",
        io_runner=lane,
        clock=lambda: clock[0],
    )
    return worker, lane, clock, source, destination, config_path


def test_output_failure_preserves_input_and_retries_later(export_lane, caplog):
    worker, lane, clock, source, destination, _ = export_lane
    destination.write_text("unavailable")
    worker.run_once()
    # Pending filesystem IO only requires a nonblocking poll.
    worker.run_once()
    assert source.is_file()
    lane.released = True
    worker.run_once()
    assert "pipeline.db_export.error" in caplog.text
    assert source.is_file()
    worker.run_once()
    assert lane.operation is None  # Honor retry delay.

    destination.unlink()
    clock[0] += 2
    worker.run_once()
    worker.run_once()
    assert len(list(destination.rglob("*.csv"))) == 2
    assert not source.exists()
    assert (source.parent / "exported" / source.name).is_file()


def test_timed_out_scan_is_closed_and_a_new_scan_can_recover(export_lane):
    worker, lane, clock, source, destination, _ = export_lane
    worker.run_once()
    lane.error = DeadlineExceeded("share IO stalled", uncertain=True)
    worker.run_once()
    assert lane.closed
    assert source.is_file()
    lane.error = None
    lane.released = True
    clock[0] += 30
    worker.run_once()
    worker.run_once()
    assert len(list(destination.rglob("*.csv"))) == 2
    assert not source.exists()


def test_missing_config_retries_without_raising_to_publisher(export_lane, caplog):
    worker, lane, clock, source, destination, config_path = export_lane
    payload = config_path.read_text()
    config_path.unlink()
    worker.run_once()
    lane.released = True
    worker.run_once()
    assert "pipeline.db_export.error" in caplog.text
    assert source.is_file()
    config_path.write_text(payload)
    clock[0] += 30
    worker.run_once()
    worker.run_once()
    assert len(list(destination.rglob("*.csv"))) == 2


def test_one_shot_flush_observes_backups_created_after_first_scan(export_lane):
    worker, lane, _clock, source, destination, _ = export_lane
    payload = source.read_bytes()
    source.unlink()
    worker.run_once()
    lane.released = True
    worker.run_once()
    assert not list(destination.rglob("*.csv"))
    source.write_bytes(payload)
    worker.flush()
    assert len(list(destination.rglob("*.csv"))) == 2
    worker.close()
    assert lane.closed
