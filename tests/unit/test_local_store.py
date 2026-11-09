import json

import pytest

from app.infrastructure import local_store
from app.infrastructure.local_store import ControllerLock, JobState, LocalJobStore


def job():
    return JobState("20261109120000", "/input", "/backup", 1.0, 7.0, 6.0, {})


def test_replay_repairs_snapshot_after_crash_between_event_and_snapshot(
    tmp_path, monkeypatch
):
    store = LocalJobStore(tmp_path)
    state = job()
    store.save(state, "discovered")
    original = local_store.atomic_json

    def fail_snapshot(path, data):
        if path.name == "job_state.json":
            raise OSError("simulated crash")
        original(path, data)

    monkeypatch.setattr(local_store, "atomic_json", fail_snapshot)
    state.status = "PUBLISHING"
    state.publications["board.csv"] = {"status": "PUBLISHING"}
    with pytest.raises(OSError):
        store.save(state, "publication.started")
    monkeypatch.setattr(local_store, "atomic_json", original)
    restored = store.load()[state.job_id]
    assert restored.status == "PUBLISHING" and restored.revision == 2
    assert (
        json.loads((store.folder(state.job_id) / "job_state.json").read_text())[
            "revision"
        ]
        == 2
    )


def test_corrupt_journal_fails_closed_instead_of_rescanning(tmp_path):
    store = LocalJobStore(tmp_path)
    state = job()
    store.save(state, "discovered")
    next(store.folder(state.job_id).glob("events/*.json")).write_text("broken")
    with pytest.raises(ValueError):
        store.load()


def test_second_controller_cannot_own_same_state(tmp_path):
    with ControllerLock(tmp_path):
        with pytest.raises(RuntimeError, match="Another controller"):
            with ControllerLock(tmp_path):
                pass
    with ControllerLock(tmp_path):
        pass


@pytest.mark.parametrize("value", ["../escape", "20261109", "20261109120000/../bad"])
def test_state_path_rejects_untrusted_job_ids(tmp_path, value):
    with pytest.raises(ValueError):
        LocalJobStore(tmp_path).folder(value)
