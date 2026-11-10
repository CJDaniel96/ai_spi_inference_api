"""Cross-platform filesystem checks for Windows deployment behavior."""

import os
import stat

import pytest


@pytest.fixture
def writable_file_fsync(monkeypatch):
    """Enforce Windows' writable-handle requirement while keeping real fsync."""
    original = os.fsync

    def fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            # A zero-byte write checks access without altering file contents.
            # Read-only descriptors raise EBADF on POSIX as well as Windows.
            os.write(fd, b"")
        return original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
