"""Await supervised IO without blocking the controller."""

from __future__ import annotations

import asyncio

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
