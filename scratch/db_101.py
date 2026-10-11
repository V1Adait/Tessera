"""Postgres 101 with asyncpg. Run from the repo root:  uv run python -m scratch.db_101

Needs DATABASE_URL in .env, e.g.  DATABASE_URL=postgresql://tessera:YOURPASSWORD@localhost:5432/tessera
"""

import asyncio

import asyncpg

from backend.app.config import get_settings


async def main() -> None:
    url = get_settings().database_url
    if not url:
        raise SystemExit("set DATABASE_URL in .env first")

    # 1. CONNECT. A connection is one open conversation with the database.
    conn = await asyncpg.connect(url, timeout=5)
    try:
        # 2. CREATE a table (a spreadsheet with typed columns).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hello (
                id         BIGSERIAL PRIMARY KEY,
                name       TEXT        NOT NULL,
                score      INTEGER     NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        # 3. INSERT. $1, $2 are placeholders: the SQL and the DATA travel separately,
        #    so data can never become SQL (this is what stops SQL injection).
        #    NEVER build SQL with f-strings from user input.
        await conn.execute("INSERT INTO hello (name, score) VALUES ($1, $2)", "ada", 42)
        await conn.execute(
            "INSERT INTO hello (name, score) VALUES ($1, $2)", "grace", 7
        )

        # 4. SELECT with a filter and an order. fetch() returns a list of Records.
        rows = await conn.fetch(
            "SELECT id, name, score FROM hello WHERE score > $1 ORDER BY score DESC", 5
        )
        for row in rows:
            print(row["id"], row["name"], row["score"])  # columns by name

        # 5. CLEAN UP so the demo can be re-run.
        await conn.execute("DROP TABLE hello")
    finally:
        await conn.close()  # always release the connection


asyncio.run(main())
