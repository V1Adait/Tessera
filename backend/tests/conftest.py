"""Fixtures shared by the test files."""

from collections.abc import AsyncIterator
from uuid import uuid4

import asyncpg
import pytest
from pydantic import ValidationError

from backend.app.config import get_settings
from backend.app.db.migrate import migrate


def _database_url() -> str | None:
    try:
        return get_settings().database_url
    except ValidationError:  # no usable .env (for example in CI): the tests simply skip
        return None


@pytest.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    """A pool on a throwaway schema with all migrations applied; dropped afterwards.

    The schema name is generated here (hex), never user input, which is the only reason it may
    be placed in SQL text. Tests that use this fixture skip when no database is reachable.
    """
    url = _database_url()
    if not url:
        pytest.skip("DATABASE_URL is not set")

    schema = f"test_{uuid4().hex[:12]}"
    try:
        admin = await asyncpg.connect(url, timeout=5)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")

    await admin.execute(f'CREATE SCHEMA "{schema}"')
    test_pool = await asyncpg.create_pool(
        url, min_size=1, max_size=5, server_settings={"search_path": schema}
    )
    assert test_pool is not None
    await migrate(test_pool)
    try:
        yield test_pool
    finally:
        await test_pool.close()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
