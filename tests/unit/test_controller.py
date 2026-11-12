"""Source markers, flat backups, bounded delivery and JSON reporting."""

import asyncio
import csv
import json
from pathlib import Path

import pytest

from app.application import controller as module
from app.application.controller import Controller
from app.infrastructure.output.sinic_csv_output import PublishCompletionUnknown
from app.tools.spi_statistics import delivery_rows, job_rows
from app.tools.spi_statistics import main as statistics_main
from tests.helpers import AsyncTestIO, FakeRuntime, make_config, write_job


def run_once(config, runtime=None):
    runtime = runtime or FakeRuntime()
    runs = asyncio.run(
        Controller(config, runtime, io_factory=AsyncTestIO).run(once=True)
    )
    return list(runs.values()), runtime


def codes(config):
    values = []
    for path in sorted(Path(config.paths.external_output_root).glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            values.extend(int(row["is_pass"]) for row in csv.DictReader(handle))
    return values


async def until(condition):
    async with asyncio.timeout(8):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.mark.usefixtures("writable_file_fsync")
def test_existing_old_date_csv_is_processed_backed_up_marked_and_exported(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, job_id="20200101120000", rows=2)
    original_path = next(source.glob("*.csv"))
    original = original_path.read_bytes()
    runs, runtime = run_once(config)
    job = runs[0]
    assert job.status == "DONE" and codes(config) == [22, 22]
    assert runtime.calls == 1 and runtime.closed
    assert not original_path.exists()
    marker = source / f"{original_path.stem}_DONE.csv"
    assert marker.read_bytes() == original
    assert (Path(job.backup_folder) / original_path.name).read_bytes() == original
    assert job.export_status == "complete"
    assert len(list(Path(config.minifi.output_path).rglob("*.csv"))) == 2
    report = json.loads(
        (Path(job.backup_folder) / "ai_result" / "run.json").read_text()
    )
    assert report["run_id"] == job.run_id
    assert report["processing_started_at"] <= report["inference_started_at"]
    record = report["publications"][original_path.name]
    assert (
        report["inference_finished_at"] <= record["returned_at"] <= record["marked_at"]
    )
    assert report["timestamps"]["processing_started_at"]
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert "state_root" not in type(config.paths).model_fields


def test_done_csvs_are_excluded_from_all_ingest_and_inference_paths(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    completed = sorted(source.glob("*.csv"))[0]
    completed.rename(completed.with_name(completed.stem + "_done.CSV"))
    runs, runtime = run_once(config)
    assert runtime.calls == 1
    assert runs[0].csv_names == (f"{source.name}_board1.csv",)
    assert codes(config) == [22]
    runs, runtime = run_once(config)
    assert runs == [] and runtime.calls == 0


def test_remove_done_after_restart_reuses_flat_backup_and_updates_latest_result(
    tmp_path,
):
    config = make_config(tmp_path)
    source = write_job(config)
    first = run_once(config)[0][0]
    first_report = (Path(first.backup_folder) / "ai_result" / "run.json").read_bytes()
    first_names = {p.name for p in Path(config.minifi.output_path).rglob("*.csv")}
    marker = next(source.glob("*_DONE.csv"))
    marker.rename(marker.with_name(marker.stem[:-5] + marker.suffix))
    second = run_once(config)[0][0]
    assert first.run_id != second.run_id and first.backup_folder == second.backup_folder
    report_path = Path(second.backup_folder) / "ai_result" / "run.json"
    assert report_path.read_bytes() != first_report
    assert json.loads(report_path.read_text())["run_id"] == second.run_id
    assert (
        len(
            {p.name for p in Path(config.minifi.output_path).rglob("*.csv")}
            - first_names
        )
        == 2
    )
    root = Path(config.paths.backup_output_root)
    assert len(job_rows(root)) == len(delivery_rows(root)) == 1


def test_restart_ignores_even_broken_reports_and_uses_only_source_markers(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    job = run_once(config)[0][0]
    (Path(job.backup_folder) / "ai_result" / "run.json").write_text("broken old report")
    runs, runtime = run_once(config)
    assert runs == [] and runtime.calls == 0
    marker = next(source.glob("*_DONE.csv"))
    marker.rename(marker.with_name(marker.stem[:-5] + marker.suffix))
    runs, runtime = run_once(config)
    assert len(runs) == runtime.calls == 1


@pytest.mark.parametrize("mode", ["slow", "failure", "missing_images"])
def test_protection_result_is_returned_and_marked_done(tmp_path, mode):
    config = make_config(tmp_path, inference_timeout=0.04)
    source = write_job(config, images=0 if mode == "missing_images" else None)
    runtime = FakeRuntime(delay=2 if mode == "slow" else 0, fail=mode == "failure")
    job = run_once(config, runtime)[0][0]
    assert job.status == "DONE" and codes(config) == [23]
    assert next(source.glob("*_DONE.csv"))
    manifest = json.loads(Path(job.result_manifest_path).read_text())
    assert manifest["outcome"] == "fallback"


@pytest.mark.parametrize(
    "images,expected_calls,expected_code", [(60, 1, 22), (61, 0, 23)]
)
def test_image_limit_counts_actual_folder_images(
    tmp_path, images, expected_calls, expected_code
):
    config = make_config(tmp_path)
    write_job(config, images=images)
    runs, runtime = run_once(config)
    assert runs[0].image_count == images
    assert runtime.calls == expected_calls and codes(config) == [expected_code]


def test_bad_csv_never_produces_fabricated_result_or_done_marker(tmp_path):
    config = make_config(tmp_path, input_timeout=0.08)
    source = write_job(config)
    next(source.glob("*.csv")).write_text("broken\n1\n")
    runs, runtime = run_once(config)
    assert runs[0].status == "INPUT_FAILED" and runtime.calls == 0
    assert codes(config) == [] and not list(source.glob("*_DONE.csv"))
    assert (Path(runs[0].backup_folder) / "ai_result" / "run.json").is_file()


def test_delivery_retries_reuse_result_and_mark_only_successful_csvs(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    original = module.publish_csv
    attempts = []

    def publish(config, frozen, expected_hash, deadline, *, job_id):
        attempts.append((frozen.name, frozen.read_bytes()))
        if "board0" in frozen.name:
            raise PermissionError("return share denied")
        return original(config, frozen, expected_hash, deadline, job_id=job_id)

    monkeypatch.setattr(module, "publish_csv", publish)
    job = run_once(config)[0][0]
    assert job.status == "PARTIAL_RETURN"
    assert (source / f"{source.name}_board0.csv").exists()
    assert (source / f"{source.name}_board1_DONE.csv").exists()
    failed = [payload for name, payload in attempts if "board0" in name]
    assert len(failed) == config.delivery.max_attempts and len(set(failed)) == 1
    assert len(list(Path(config.minifi.output_path).rglob("*.csv"))) == 2


def test_unknown_delivery_is_not_retried_or_marked(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    source = write_job(config)
    calls = []

    def unknown(*args, **kwargs):
        calls.append(1)
        raise PublishCompletionUnknown("rename acknowledgement lost")

    monkeypatch.setattr(module, "publish_csv", unknown)
    job = run_once(config)[0][0]
    assert job.status == "PUBLICATION_UNKNOWN" and len(calls) == 1
    assert not list(source.glob("*_DONE.csv"))
    assert job.export_status == "disabled"


def test_marker_failure_retries_only_rename_not_inference_or_delivery(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    write_job(config)
    original = module.publish_csv
    deliveries = []

    def publish(*args, **kwargs):
        deliveries.append(1)
        return original(*args, **kwargs)

    def fail(*args, **kwargs):
        raise PermissionError("source rename denied")

    monkeypatch.setattr(module, "publish_csv", publish)
    monkeypatch.setattr(module, "mark_done", fail)
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)
    job = next(iter(asyncio.run(controller.run(once=True)).values()))
    assert job.status == "MARK_PENDING"
    assert len(deliveries) == runtime.calls == 1
    assert next(iter(job.publications.values()))["marker_error"]


def test_done_collision_and_changed_source_are_never_overwritten(tmp_path):
    from app.application.operations import file_hash, mark_done

    source = tmp_path / "board.csv"
    source.write_bytes(b"original")
    digest = file_hash(source)
    marker = tmp_path / "board_DONE.csv"
    marker.write_bytes(b"older result")
    with pytest.raises(FileExistsError):
        mark_done(source, digest)
    assert marker.read_bytes() == b"older result" and source.exists()
    marker.unlink()
    source.write_bytes(b"new data")
    with pytest.raises(ValueError, match="changed"):
        mark_done(source, digest)
    assert source.read_bytes() == b"new data" and not marker.exists()


def test_continuous_scanning_detects_new_csv_and_live_done_removal(tmp_path):
    config = make_config(tmp_path)
    Path(config.paths.watch_root).mkdir(parents=True)
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(lambda: controller._discovery_lane is not None)
            source = write_job(config)
            await until(
                lambda: bool(list(source.glob("*_DONE.csv"))) and not controller.jobs
            )
            marker = next(source.glob("*_DONE.csv"))
            marker.rename(marker.with_name(marker.stem[:-5] + marker.suffix))
            await until(
                lambda: (
                    runtime.calls == 2
                    and bool(list(source.glob("*_DONE.csv")))
                    and not controller.jobs
                )
            )
            # Keep the original folder, then add another CSV with its image.
            csv_path = next(source.glob("*_DONE.csv"))
            extra = source / "extra.csv"
            extra.write_bytes(csv_path.read_bytes())
            image = next(source.glob("*.jpg"))
            (source / "extra_J001_001_0.jpg").write_bytes(image.read_bytes())
            await until(
                lambda: (source / "extra_DONE.csv").exists() and not controller.jobs
            )
            assert runtime.calls == 3
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    assert (
        len(list(Path(config.paths.backup_output_root).glob("*/*/ai_result/run.json")))
        == 1
    )


def test_minifi_failure_does_not_resend_and_can_retry_from_committed_result(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    write_job(config)
    original = module.export_result
    attempts = []

    def transient(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("output unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "export_result", transient)
    runs, runtime = run_once(config)
    assert runs[0].export_status == "complete" and len(attempts) == 2
    assert runtime.calls == 1
    assert len(next(iter(runs[0].publications.values()))["attempt_log"]) == 1


def test_statistics_reads_json_after_backup_is_moved(tmp_path):
    config = make_config(tmp_path)
    write_job(config)
    run_once(config)
    original = Path(config.paths.backup_output_root)
    moved = tmp_path / "moved-backup"
    original.rename(moved)
    config.paths.backup_output_root = str(moved)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    assert (
        statistics_main(
            ["--config", str(config_path), "--output", str(tmp_path / "reports")]
        )
        == 0
    )
    with (tmp_path / "reports/jobs.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["pass_count"] == "1" and row["marked_csv_count"] == "1"


def test_known_delivery_failure_is_retried_by_next_scan_until_done(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    config.delivery.max_attempts = 1
    config.runtime.background_retry_seconds = 0.04
    source = write_job(config)
    original = module.publish_csv
    calls = []

    def transient(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("temporary denial")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "publish_csv", transient)
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(
                lambda: bool(list(source.glob("*_DONE.csv"))) and not controller.jobs
            )
            assert len(calls) == runtime.calls == 2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


def test_unknown_delivery_is_suppressed_across_scans_but_restart_accepts_unmarked_csv(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    config.runtime.background_retry_seconds = 0.02
    source = write_job(config)
    calls = []

    def unknown(*args, **kwargs):
        calls.append(1)
        raise PublishCompletionUnknown("lost acknowledgement")

    monkeypatch.setattr(module, "publish_csv", unknown)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(lambda: len(calls) == 1 and not controller.jobs)
            await asyncio.sleep(0.15)
            assert len(calls) == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    assert not list(source.glob("*_DONE.csv"))
    assert len(run_once(config)[0]) == 1 and len(calls) == 2


def test_marker_retry_completes_json_without_extra_delivery(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.runtime.background_retry_seconds = 0.02
    source = write_job(config)
    original = module.mark_done
    marker_calls = []

    def transient(*args, **kwargs):
        marker_calls.append(1)
        if len(marker_calls) <= 2:
            raise PermissionError("source temporarily locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "mark_done", transient)
    controller = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(
                lambda: (
                    bool(list(source.glob("*_DONE.csv")))
                    and not controller.jobs
                    and not controller._background_runs
                )
            )
            report = json.loads(
                next(
                    Path(config.paths.backup_output_root).glob("*/*/ai_result/run.json")
                ).read_text()
            )
            assert report["status"] == "DONE"
            record = next(iter(report["publications"].values()))
            assert record["marker_attempts"] == 3
            assert len(record["attempt_log"]) == controller.runtime.calls == 1
            assert not list(controller.artifacts.iterdir())
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


def test_csv_arriving_during_inference_is_left_for_next_run(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)

    class LateCsvRuntime(FakeRuntime):
        async def infer(self, images, *, deadline_at):
            if self.calls == 0:
                original = next(source.glob("*.csv"))
                (source / "late.csv").write_bytes(original.read_bytes())
                image = next(source.glob("*.jpg"))
                (source / "late_J001_001_0.jpg").write_bytes(image.read_bytes())
            return await super().infer(images, deadline_at=deadline_at)

    runtime = LateCsvRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(
                lambda: (source / "late_DONE.csv").exists() and not controller.jobs
            )
            assert runtime.calls == 2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    reports = [
        json.loads(p.read_text())
        for p in Path(config.paths.backup_output_root).glob("*/*/ai_result/run.json")
    ]
    assert len(reports) == 1 and reports[0]["csv_names"] == ["late.csv"]
    assert len(codes(config)) == 2


def test_lock_is_held_until_workers_have_finished_closing(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    Path(config.paths.watch_root).mkdir(parents=True)
    first = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
    original_close = first.close

    async def exercise():
        closing = asyncio.Event()
        release = asyncio.Event()

        async def delayed_close():
            closing.set()
            await release.wait()
            await original_close()

        monkeypatch.setattr(first, "close", delayed_close)
        task = asyncio.create_task(first.run())
        await until(lambda: first._discovery_lane is not None)
        task.cancel()
        await closing.wait()
        try:
            second = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
            with pytest.raises(RuntimeError, match="Another controller"):
                await second.run(once=True)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        third = Controller(config, FakeRuntime(), io_factory=AsyncTestIO)
        assert await third.run(once=True) == {}

    asyncio.run(exercise())


def test_csv_settling_does_not_make_unknown_delivery_look_like_another_new_file(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    config.runtime.background_retry_seconds = 0.02
    source = write_job(config)
    calls = []

    class EditingWarmup(FakeRuntime):
        async def start(self):
            await until(lambda: bool(controller.jobs))
            next(source.glob("*.csv")).touch()

    runtime = EditingWarmup()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)

    def unknown(*args, **kwargs):
        calls.append(1)
        raise PublishCompletionUnknown("uncertain delivery")

    monkeypatch.setattr(module, "publish_csv", unknown)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(lambda: len(calls) == 1 and not controller.jobs)
            await asyncio.sleep(0.15)
            assert len(calls) == runtime.calls == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


def test_csv_changed_between_stability_check_and_snapshot_is_waited_for_again(
    tmp_path, monkeypatch
):
    from app.application.ingest import io as ingest_io

    config = make_config(tmp_path)
    source = write_job(config)
    original = ingest_io.snapshot_csv
    calls = []

    def changed(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            next(source.glob("*.csv")).touch()
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_io, "snapshot_csv", changed)
    runs, runtime = run_once(config)
    assert len(calls) == 2
    assert runs[0].status == "DONE" and runtime.calls == 1
    assert next(source.glob("*_DONE.csv"))


@pytest.mark.usefixtures("writable_file_fsync")
def test_backup_timestamp_folder_contains_flat_originals_and_ai_result_only(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, rows=2)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    job = run_once(config)[0][0]
    backup = Path(config.paths.backup_output_root) / "2026-11-09" / source.name
    assert Path(job.backup_folder) == backup
    assert {p.name for p in backup.iterdir()} == set(originals) | {"ai_result"}
    for name, payload in originals.items():
        assert (backup / name).read_bytes() == payload
    assert (backup / "ai_result/run.json").is_file()
    assert (backup / "ai_result/manifest.json").is_file()
    assert not (backup / job.run_id).exists()
    assert not Path(job.workspace_folder).exists()
    assert not list(backup.rglob("*.tmp"))


def test_flat_backup_with_existing_ai_result_and_originals_is_reused(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    backup = Path(config.paths.backup_output_root) / "2026-11-09" / source.name
    (backup / "ai_result").mkdir(parents=True)
    (backup / "ai_result/run.json").write_text("older report")
    for p in source.iterdir():
        (backup / p.name).write_bytes(p.read_bytes())
    job = run_once(config)[0][0]
    assert job.status == "DONE" and job.reason is None
    assert next(source.glob("*_DONE.csv"))


def test_changed_source_image_does_not_overwrite_flat_original_backup(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config)
    first = run_once(config)[0][0]
    image = next(source.glob("*.jpg"))
    original = (Path(first.backup_folder) / image.name).read_bytes()
    image.write_bytes(original + b"changed source bytes")
    marker = next(source.glob("*_DONE.csv"))
    marker.rename(marker.with_name(marker.stem[:-5] + marker.suffix))
    jobs, runtime = run_once(config)
    assert jobs[0].status == "INPUT_FAILED"
    assert jobs[0].reason == "raw_backup_conflict" and runtime.calls == 0
    assert (Path(first.backup_folder) / image.name).read_bytes() == original
    assert not list(source.glob("*_DONE.csv"))


def test_rerun_replaces_selected_csv_result_and_keeps_other_csv_results(tmp_path):
    config = make_config(tmp_path)
    source = write_job(config, csv_count=2)
    first = run_once(config)[0][0]
    root = Path(first.backup_folder)
    other_name = f"{source.name}_board0.csv"
    rerun_name = f"{source.name}_board1.csv"
    other_result = (root / "ai_result/returned" / other_name).read_bytes()
    marker = source / f"{source.name}_board1_DONE.csv"
    marker.rename(source / rerun_name)
    second = run_once(config, FakeRuntime(fail=True))[0][0]
    assert second.status == "DONE" and second.run_id != first.run_id
    assert list(second.publications) == [rerun_name]
    assert (root / "ai_result/returned" / other_name).read_bytes() == other_result
    with (root / "ai_result/returned" / rerun_name).open(
        encoding="utf-8-sig"
    ) as handle:
        assert [row["is_pass"] for row in csv.DictReader(handle)] == ["23"]
    assert (
        root / "ai_result/exported" / f"{Path(rerun_name).stem}_processed.csv"
    ).exists()
    assert not (
        root / "ai_result/processed" / f"{Path(rerun_name).stem}_processed.csv"
    ).exists()


def test_new_batch_statistics_do_not_use_previous_batch_manifest(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    source = write_job(config)
    first = run_once(config)[0][0]
    marker = next(source.glob("*_DONE.csv"))
    marker.rename(marker.with_name(marker.stem[:-5] + marker.suffix))
    checked = []
    original = module.write_report

    def observe(run):
        original(run)
        if run.status == "QUEUED":
            row = job_rows(Path(config.paths.backup_output_root))[0]
            assert row["run_id"] != first.run_id
            assert row["outcome"] == "" and row["pass_count"] == 0
            checked.append(True)

    monkeypatch.setattr(module, "write_report", observe)
    assert run_once(config)[0][0].status == "DONE"
    assert checked


def test_flat_backup_accepts_configured_image_extension_without_leading_dot(tmp_path):
    config = make_config(tmp_path)
    config.processing.image_extensions = ["JPG"]
    source = write_job(config)
    job = run_once(config)[0][0]
    image = next(source.glob("*.jpg"))
    assert job.status == "DONE"
    assert (Path(job.backup_folder) / image.name).read_bytes() == image.read_bytes()


def test_same_folder_waits_for_pending_export_while_other_folders_continue(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    config.runtime.background_retry_seconds = 0.02
    source = write_job(config)
    original_export = module.export_result
    gate = {"ready": False}

    def export(settings, folder, names):
        if folder.name == source.name and not gate["ready"]:
            raise OSError("temporary MiNiFi failure")
        return original_export(settings, folder, names)

    monkeypatch.setattr(module, "export_result", export)
    runtime = FakeRuntime()
    controller = Controller(config, runtime, io_factory=AsyncTestIO)

    async def exercise():
        task = asyncio.create_task(controller.run())
        try:
            await until(
                lambda: bool(controller._background_runs) and not controller.jobs
            )
            csv_path = next(source.glob("*_DONE.csv"))
            (source / "extra.csv").write_bytes(csv_path.read_bytes())
            image = next(source.glob("*.jpg"))
            (source / "extra_J001_001_0.jpg").write_bytes(image.read_bytes())
            other = write_job(config, job_id="20261109120100")
            await until(
                lambda: bool(list(other.glob("*_DONE.csv"))) and not controller.jobs
            )
            assert runtime.calls == 2
            assert (source / "extra.csv").exists()
            assert not (source / "extra_DONE.csv").exists()
            gate["ready"] = True
            await until(
                lambda: (source / "extra_DONE.csv").exists() and not controller.jobs
            )
            assert runtime.calls == 3
            report = json.loads(
                (
                    Path(config.paths.backup_output_root)
                    / "2026-11-09"
                    / source.name
                    / "ai_result/run.json"
                ).read_text()
            )
            assert report["csv_names"] == ["extra.csv"]
            assert report["export_status"] == "complete"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


def test_crop_setting_is_recorded_while_backup_keeps_full_original_image(tmp_path):
    import cv2

    config = make_config(tmp_path)
    config.inference.obb.center_crop_ratio = 0.6
    source = write_job(config)
    image = next(source.glob("*.jpg"))
    original = image.read_bytes()
    job = run_once(config)[0][0]
    backup_image = Path(job.backup_folder) / image.name
    assert backup_image.read_bytes() == image.read_bytes() == original
    assert cv2.imread(str(backup_image)).shape[:2] == (4, 5)
    report = json.loads((Path(job.backup_folder) / "ai_result/run.json").read_text())
    assert report["config_snapshot"]["inference"]["obb"]["center_crop_ratio"] == 0.6
    manifest = json.loads(Path(job.result_manifest_path).read_text())
    assert manifest["inspection_metadata"]["preprocessing"]["center_crop_ratio"] == 0.6
