"""Unit tests for the scanner retry/backoff/dead-letter policy."""

from __future__ import annotations

from scan_jobs import FailureTracker, backoff_seconds, classify_status


def test_sinic_input_mode_does_not_require_done_file(tmp_path) -> None:
    from datetime import date

    from app.infrastructure.input.sinic_folder_input import (
        FileSettleTracker,
        SinicFolderInput,
    )

    job_dir = tmp_path / "20260821112233"
    job_dir.mkdir()
    (job_dir / "result.CSV").write_text("is_pass\n1\n", encoding="utf-8")
    (job_dir / "image.jpg").write_bytes(b"image")

    adapter = SinicFolderInput()
    candidates = adapter.list_candidates(tmp_path, date(2026, 8, 21))
    tracker = FileSettleTracker(settle_seconds=0)

    assert candidates == (job_dir,)
    assert tracker.observe(adapter.load(job_dir), now=1.0) is False
    assert tracker.observe(adapter.load(job_dir), now=1.0) is True


def test_classify_status() -> None:
    assert classify_status(200) == "success"
    assert classify_status(204) == "success"
    assert classify_status(400) == "client_error"
    assert classify_status(404) == "client_error"
    assert classify_status(500) == "server_error"
    assert classify_status(-1) == "server_error"


def test_backoff_is_exponential_and_capped() -> None:
    assert backoff_seconds(1, 5.0, 300.0) == 5.0
    assert backoff_seconds(2, 5.0, 300.0) == 10.0
    assert backoff_seconds(3, 5.0, 300.0) == 20.0
    assert backoff_seconds(20, 5.0, 300.0) == 300.0  # capped
    assert backoff_seconds(0, 5.0, 300.0) == 0.0


def test_unknown_job_is_not_skipped() -> None:
    tracker = FailureTracker()
    assert tracker.should_skip("k", now=0.0) == (False, "")


def test_client_error_dead_letters_quickly() -> None:
    tracker = FailureTracker(client_error_max_attempts=2, max_retries=5)

    attempts, dead = tracker.record_failure("k", 400, now=0.0)
    assert (attempts, dead) == (1, False)

    attempts, dead = tracker.record_failure("k", 400, now=100.0)
    assert dead is True  # dead-lettered at the 2nd client-error attempt
    assert tracker.is_dead("k")
    assert tracker.should_skip("k", now=10_000.0) == (True, "dead")


def test_server_error_backs_off_then_dead_letters() -> None:
    tracker = FailureTracker(
        max_retries=3, base_backoff_seconds=5.0, max_backoff_seconds=300.0
    )

    _, dead = tracker.record_failure("k", 500, now=0.0)
    assert dead is False
    assert tracker.should_skip("k", now=1.0) == (True, "backoff")  # within backoff
    assert tracker.should_skip("k", now=1000.0) == (False, "")  # backoff elapsed

    tracker.record_failure("k", 500, now=1000.0)
    _, dead = tracker.record_failure("k", 500, now=2000.0)
    assert dead is True  # 3rd attempt reaches max_retries


def test_transport_error_treated_as_server_error() -> None:
    tracker = FailureTracker(max_retries=5)
    _, dead = tracker.record_failure("k", -1, now=0.0)
    assert dead is False
    assert tracker.should_skip("k", now=0.0) == (True, "backoff")


def test_success_clears_failure_state() -> None:
    tracker = FailureTracker()
    tracker.record_failure("k", 500, now=0.0)
    assert tracker.should_skip("k", now=0.0)[0] is True

    tracker.record_success("k")
    assert tracker.should_skip("k", now=0.0) == (False, "")
