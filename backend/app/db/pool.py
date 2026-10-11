"""The connection pool: a small set of open database connections shared by all requests.

Opening a connection is slow, so we open a few once at startup and lend them out.
Use it as:   async with pool.acquire() as conn: ...
"""

import asyncpg
from fastapi import HTTPException, Request

from backend.app.config import Settings


async def create_pool(
    settings: Settings, *, search_path: str | None = None
) -> asyncpg.Pool:
    """Open the pool. `search_path` lets tests work inside their own throwaway schema."""
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set")

    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=1,
        max_size=10,
        timeout=5,  # seconds to open a connection: fail instead of hanging
        command_timeout=10,  # seconds any single query may take
        server_settings={"search_path": search_path} if search_path else None,
    )
    if pool is None:  # only happens with the pool's own "no-op" mode; keeps mypy honest
        raise RuntimeError("could not create the database pool")
    return pool


def get_pool(request: Request) -> asyncpg.Pool:
    """FastAPI dependency: the pool opened at startup (see main.py lifespan)."""
    pool: asyncpg.Pool | None = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database is not configured")
    return pool
