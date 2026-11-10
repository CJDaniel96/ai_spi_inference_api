"""Keep cancelled async callers from hiding outstanding local file writers."""

from __future__ import annotations

import asyncio


class LocalFileTasks:
    def __init__(self):
        self._tasks: dict[str, set[asyncio.Task]] = {}

    async def run(self, job_id, function, /, *args, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._tasks.setdefault(job_id, set()).add(task)

        def finished(task):
            pending = self._tasks[job_id]
            pending.discard(task)
            if not pending:
                self._tasks.pop(job_id)
            if not task.cancelled():
                task.exception()

        task.add_done_callback(finished)
        # Cancelling to_thread does not stop its thread. Keep the inner task
        # alive so cleanup can observe when the actual writer has finished.
        return await asyncio.shield(task)

    def busy(self, job_id):
        return any(not task.done() for task in self._tasks.get(job_id, ()))

    async def drain(self):
        tasks = [task for pending in self._tasks.values() for task in pending]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
