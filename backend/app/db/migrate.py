"""A tiny migration runner: applies migrations/*.sql in name order, once each.

    uv run python -m backend.app.db.migrate

Each file runs in its own transaction together with the bookkeeping row, so a migration is
either fully applied or not at all. Applied names are remembered in `schema_migrations`.
"""

import asyncio
from pathlib import Path

import asyncpg

from backend.app.config import get_settings
from backend.app.db.pool import create_pool

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 7_351_001  # any constant: stops two processes migrating at the same time

_CREATE_BOOKKEEPING = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        name        text        PRIMARY KEY,
        applied_at  timestamptz NOT NULL DEFAULT now()
    )
"""


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


async def pending_migrations(pool: asyncpg.Pool) -> list[str]:
    """Names of migration files that have not been applied yet (read-only)."""
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT to_regclass('schema_migrations') IS NOT NULL"
        )
        applied: set[str] = set()
        if exists:
            applied = {
                r["name"]
                for r in await conn.fetch("SELECT name FROM schema_migrations")
            }
    return [p.name for p in migration_files() if p.name not in applied]


async def migrate(pool: asyncpg.Pool) -> list[str]:
    """Apply every pending migration. Returns the names applied (empty if up to date)."""
    applied_now: list[str] = []
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1)", _LOCK_ID)
        try:
            await conn.execute(_CREATE_BOOKKEEPING)
            done = {
                r["name"]
                for r in await conn.fetch("SELECT name FROM schema_migrations")
            }
            for path in migration_files():
                if path.name in done:
                    continue
                async with conn.transaction():
                    await conn.execute(path.read_text(encoding="utf-8"))
                    await conn.execute(
                        "INSERT INTO schema_migrations (name) VALUES ($1)", path.name
                    )
                applied_now.append(path.name)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_ID)
    return applied_now


async def _main() -> None:
    pool = await create_pool(get_settings())
    try:
        applied = await migrate(pool)
    finally:
        await pool.close()
    print("applied:", ", ".join(applied) if applied else "nothing (already up to date)")


if __name__ == "__main__":
    asyncio.run(_main())
