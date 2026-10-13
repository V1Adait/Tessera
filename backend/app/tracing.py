"""Trace rows for model calls made outside a chat turn (classify, plan, extract).

`Recorder` is "something that stores a TraceRecord". Components take one as an argument, so
tests pass a list-appender and production passes `db_recorder(pool)`. Recording must never break
the request it describes: a failed insert is logged and dropped.
"""

import logging
from collections.abc import Awaitable, Callable

import asyncpg

from backend.app.db import repository as repo
from backend.app.db.records import TraceRecord

logger = logging.getLogger(__name__)

Recorder = Callable[[TraceRecord], Awaitable[None]]


async def no_recorder(trace: TraceRecord) -> None:
    """The default: record nothing."""


def db_recorder(pool: asyncpg.Pool) -> Recorder:
    async def record(trace: TraceRecord) -> None:
        try:
            await repo.record_trace(pool, trace)
        except Exception:
            logger.exception(
                "could not store trace", extra={"request_id": str(trace.request_id)}
            )

    return record
