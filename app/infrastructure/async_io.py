"""Await supervised IO without blocking the controller's deadline monitor."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.infrastructure.deadline_io import DeadlineIO


async def call_io(lane: DeadlineIO, function, *args, deadline_at: float, **kwargs):
    try:
        lane.submit(function, *args, deadline_at=deadline_at, function_kwargs=kwargs)
        while True:
            done, result = lane.poll()
            if done:
                return result
            await asyncio.sleep(0.005)
    except BaseException:
        lane.close()
        raise


class DeadlineGate:
    """Serialize publication, choosing the earliest queued deadline."""

    def __init__(self):
        self.condition = asyncio.Condition()
        self.waiting = {}
        self.owner = None

    @asynccontextmanager
    async def claim(self, job_id, deadline):
        async with self.condition:
            if job_id in self.waiting or self.owner == job_id:
                raise RuntimeError("Duplicate publication request")
            self.waiting[job_id] = deadline
            try:
                await self.condition.wait_for(
                    lambda: (
                        self.owner is None
                        and min(self.waiting, key=lambda key: (self.waiting[key], key))
                        == job_id
                    )
                )
            except BaseException:
                self.waiting.pop(job_id, None)
                self.condition.notify_all()
                raise
            self.owner = job_id
            self.waiting.pop(job_id)
        try:
            yield
        finally:
            async with self.condition:
                self.owner = None
                self.condition.notify_all()
