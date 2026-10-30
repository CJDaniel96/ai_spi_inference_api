"""Supervise blocking share IO in a reusable, killable spawned process.

Only plain data crosses the pipe. SQLite transactions/leases stay in the parent.
A timed out publication has an uncertain remote commit, never a safe retry.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any


class DeadlineExceeded(TimeoutError):
    """The absolute job budget was exhausted."""

    def __init__(self, message: str, *, uncertain: bool = False):
        super().__init__(message)
        self.uncertain = uncertain


def _serve(connection: Connection) -> None:
    try:
        while True:
            function, args, kwargs, deadline = connection.recv()
            try:
                if time.time() >= deadline:
                    raise DeadlineExceeded("IO budget exhausted before execution")
                result = function(*args, **kwargs)
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, exc))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class DeadlineIO:
    """One sequential IO lane; timeout destroys the lane before it is reused."""

    def __init__(self) -> None:
        self._process = None
        self._connection = None
        self._stop_at = None
        self._deadline_at = None
        self.last_terminated_pid = None

    def call(
        self,
        function: Callable[..., Any],
        *args: Any,
        deadline_at: float,
        function_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        self.submit(
            function,
            *args,
            deadline_at=deadline_at,
            function_kwargs=function_kwargs,
            **kwargs,
        )
        while True:
            done, result = self.poll()
            if done:
                return result
            time.sleep(0.005)

    def submit(
        self,
        function: Callable[..., Any],
        *args: Any,
        deadline_at: float,
        function_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Start one background operation without blocking the caller."""
        if self._stop_at is not None:
            raise RuntimeError("IO lane already has a pending operation")
        remaining = deadline_at - time.time()
        if remaining <= 0:
            raise DeadlineExceeded("IO budget exhausted")
        self._stop_at = time.monotonic() + remaining
        self._deadline_at = deadline_at
        try:
            if self._process is None:
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                self._connection = parent
                self._process = context.Process(
                    target=_serve, args=(child,), daemon=True
                )
                try:
                    self._process.start()
                finally:
                    child.close()
            self._connection.send(
                (function, args, {**kwargs, **(function_kwargs or {})}, deadline_at)
            )
        except BaseException:
            self.close()
            raise

    def poll(self) -> tuple[bool, Any]:
        """Return an acknowledgement, or enforce the background IO deadline."""
        try:
            if self._stop_at is None:
                return False, None
            if time.monotonic() >= self._stop_at or time.time() >= self._deadline_at:
                raise DeadlineExceeded(
                    "blocking IO reached the job deadline", uncertain=True
                )
            if self._connection.poll(0):
                success, result = self._connection.recv()
                self._stop_at = self._deadline_at = None
                if success:
                    return True, result
                raise result
            if not self._process.is_alive():
                raise BrokenPipeError("IO process exited without an acknowledgement")
            return False, None
        except (DeadlineExceeded, EOFError, BrokenPipeError):
            self.close()
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._process is not None and self._process.pid is not None:
            if self._stop_at is not None:
                self.last_terminated_pid = self._process.pid
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.05)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.05)
        self._process = None
        self._connection = None
        self._stop_at = self._deadline_at = None

    def __del__(self) -> None:
        self.close()
