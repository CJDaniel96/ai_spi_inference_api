import json

import pytest

from app.infrastructure.local_store import ControllerLock, atomic_json


def test_second_controller_cannot_own_same_input(tmp_path):
    with ControllerLock(tmp_path):
        with pytest.raises(RuntimeError, match="Another controller"):
            with ControllerLock(tmp_path):
                pass
    with ControllerLock(tmp_path):
        pass


@pytest.mark.usefixtures("writable_file_fsync")
def test_atomic_report_replaces_previous_json_without_temp_files(tmp_path):
    path = tmp_path / "backup" / "run.json"
    atomic_json(path, {"status": "PUBLISHING"})
    atomic_json(path, {"status": "DONE", "marked_at": 42})
    assert json.loads(path.read_text()) == {"status": "DONE", "marked_at": 42}
    assert list(path.parent.iterdir()) == [path]
