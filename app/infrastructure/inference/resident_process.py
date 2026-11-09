"""One warm OBB model in a spawned process, with bounded job waits."""

from __future__ import annotations

import asyncio
import multiprocessing
import time
import uuid
from multiprocessing.connection import Connection

from app.core.inspection_config import ObbConfig
from app.core.logging import get_logger
from app.domain.entities.obb_result import ObbBatchResult
from app.infrastructure.inference.yolo_obb import YoloObbModel


class ObbRuntimeError(RuntimeError):
    """The model process failed or returned an invalid response."""


def _serve(connection: Connection, settings: ObbConfig, model_factory) -> None:
    try:
        try:
            model = model_factory(settings)
            connection.send(
                (
                    "ready",
                    {
                        "model_version": model.model_version,
                        "device": model.device,
                    },
                )
            )
        except Exception as exc:
            connection.send(("error", f"OBB startup failed: {exc}"))
            return
        while True:
            request_id, images, deadline_at = connection.recv()
            try:
                if time.time() >= deadline_at:
                    raise TimeoutError("OBB task expired before execution")
                result = model.infer(images, deadline_at)
                if time.time() >= deadline_at:
                    raise TimeoutError("OBB result arrived after the deadline")
                connection.send((request_id, result))
            except Exception as exc:
                connection.send(("error", f"OBB inference failed: {exc}"))
                # A backend error can invalidate its CUDA context. Recreate it
                # for the next job rather than trusting the failed instance.
                return
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class ResidentObbRuntime:
    """Single-flight IPC; neither model objects nor GPU tensors leave the child."""

    def __init__(self, settings: ObbConfig, *, model_factory=YoloObbModel) -> None:
        self._settings = settings
        self._factory = model_factory
        self._process = None
        self._connection = None
        self._ready = False
        self._busy = False
        self.metadata = {}
        self._logger = get_logger("controller.obb")

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    async def _receive(self, deadline_at: float):
        stop_at = time.monotonic() + max(0, deadline_at - time.time())
        while True:
            if time.time() >= deadline_at or time.monotonic() >= stop_at:
                raise TimeoutError("Resident OBB process exceeded its deadline")
            if self._connection.poll(0):
                return self._connection.recv()
            if not self._process.is_alive():
                raise ObbRuntimeError("Resident OBB process exited without a response")
            await asyncio.sleep(0.005)

    async def start(self, *, deadline_at: float | None = None) -> None:
        if self._ready and self._process is not None and self._process.is_alive():
            return
        self.close()
        deadline = min(
            deadline_at if deadline_at is not None else float("inf"),
            time.time() + self._settings.startup_timeout_seconds,
        )
        if deadline <= time.time():
            raise TimeoutError("No budget remains to start the OBB model")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_serve, args=(child, self._settings, self._factory), daemon=True
        )
        try:
            try:
                self._process.start()
            finally:
                child.close()
            kind, payload = await self._receive(deadline)
            if kind != "ready":
                raise ObbRuntimeError(str(payload))
            self.metadata = dict(payload)
            self._ready = True
            self._logger.info(
                "event=obb.ready pid=%s metadata=%s", self.pid, self.metadata
            )
        except BaseException:
            self.close()
            raise

    async def infer(
        self, images: dict[str, str], *, deadline_at: float
    ) -> ObbBatchResult:
        if self._busy:
            raise ObbRuntimeError("Resident OBB process already has an active task")
        self._busy = True
        try:
            await self.start(deadline_at=deadline_at)
            request_id = uuid.uuid4().hex
            self._connection.send((request_id, dict(images), deadline_at))
            response_id, payload = await self._receive(deadline_at)
            if response_id != request_id or not isinstance(payload, ObbBatchResult):
                raise ObbRuntimeError(f"Invalid OBB response: {response_id}: {payload}")
            return payload
        except BaseException:
            self.close()
            raise
        finally:
            self._busy = False

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._process is not None and self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.1)
        self._connection = None
        self._process = None
        self._ready = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_exc):
        self.close()
