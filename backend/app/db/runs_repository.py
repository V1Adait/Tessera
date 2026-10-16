"""SQL for matrix runs: the book-keeping of a batch fill. Same conventions as repository.py."""

import json
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg
from pydantic import BaseModel

from backend.app.db import repository as repo

Db = repo.Db


class RunInProgressError(Exception):
    """This workspace already has a run going. Two at once would fight over the same cells."""


class RunRecord(BaseModel):
    id: UUID
    workspace_id: UUID
    status: str
    scope: dict[str, Any]
    total: int
    done: int
    failed: int
    skipped: int
    spent_usd: Decimal
    unknown_cost_calls: int
    cap_usd: Decimal
    estimate_usd: Decimal | None
    detail: str | None
    created_at: Any
    finished_at: Any


_COLUMNS = """
    id, workspace_id, status, scope, total, done, failed, skipped, spent_usd, unknown_cost_calls,
    cap_usd, estimate_usd, detail, created_at, finished_at
"""


def _run(row: asyncpg.Record) -> RunRecord:
    data = dict(row)
    scope = data["scope"]
    data["scope"] = json.loads(scope) if isinstance(scope, str) else scope
    return RunRecord(**data)


async def create_run(
    db: Db,
    workspace_id: UUID,
    scope: dict[str, Any],
    *,
    total: int,
    cap_usd: Decimal,
    estimate_usd: Decimal | None,
    status: str = "running",
) -> RunRecord:
    """Raises RunInProgressError if the workspace already has a running run."""
    try:
        row = await db.fetchrow(
            f"""
            INSERT INTO matrix_runs (workspace_id, status, scope, total, cap_usd, estimate_usd,
                                     finished_at)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, CASE WHEN $2 = 'running' THEN NULL ELSE now() END)
            RETURNING {_COLUMNS}
            """,
            workspace_id,
            status,
            json.dumps(scope),
            total,
            cap_usd,
            estimate_usd,
        )
    except asyncpg.UniqueViolationError:
        raise RunInProgressError(str(workspace_id)) from None
    assert row is not None
    return _run(row)


async def get_run(db: Db, run_id: UUID) -> RunRecord | None:
    row = await db.fetchrow(f"SELECT {_COLUMNS} FROM matrix_runs WHERE id = $1", run_id)
    return _run(row) if row else None


async def list_runs(db: Db, workspace_id: UUID, limit: int = 20) -> list[RunRecord]:
    rows = await db.fetch(
        f"""
        SELECT {_COLUMNS} FROM matrix_runs WHERE workspace_id = $1
        ORDER BY created_at DESC, id LIMIT $2
        """,
        workspace_id,
        limit,
    )
    return [_run(r) for r in rows]


async def update_progress(
    db: Db,
    run_id: UUID,
    *,
    done: int,
    failed: int,
    skipped: int,
    spent_usd: Decimal,
    unknown_cost_calls: int,
) -> None:
    await db.execute(
        """
        UPDATE matrix_runs
        SET done = $2, failed = $3, skipped = $4, spent_usd = $5, unknown_cost_calls = $6
        WHERE id = $1 AND status = 'running'
        """,
        run_id,
        done,
        failed,
        skipped,
        spent_usd,
        unknown_cost_calls,
    )


async def finish_run(
    db: Db, run_id: UUID, status: str, detail: str | None = None
) -> None:
    await db.execute(
        """
        UPDATE matrix_runs SET status = $2, detail = $3, finished_at = now()
        WHERE id = $1 AND status = 'running'
        """,
        run_id,
        status,
        detail,
    )


async def recover_interrupted_runs(db: Db) -> int:
    """At start-up: a run still marked running was cut off by a shutdown or a crash."""
    result = await db.execute(
        """
        UPDATE matrix_runs
        SET status = 'interrupted', finished_at = now(),
            detail = 'the server stopped while this run was going'
        WHERE status = 'running'
        """
    )
    return int(result.split()[-1])


# --- what planning needs ------------------------------------------------------------------------


async def cell_states(
    db: Db, workspace_id: UUID
) -> dict[tuple[UUID, UUID], tuple[str, bool, bool]]:
    """(entity_id, column_id) -> (status, stale, pinned) for every cell in the workspace."""
    rows = await db.fetch(
        "SELECT entity_id, column_id, status, stale, pinned FROM cells WHERE workspace_id = $1",
        workspace_id,
    )
    return {
        (r["entity_id"], r["column_id"]): (r["status"], r["stale"], r["pinned"])
        for r in rows
    }


async def measured_cost_per_call(
    db: Db, model: str, sample: int = 200
) -> tuple[Decimal | None, int]:
    """The average cost of this model's recent extraction calls (only calls whose cost is known),
    and how many calls that average is built from."""
    rows = await db.fetch(
        """
        SELECT cost_usd FROM traces
        WHERE kind = 'extract' AND model = $1 AND status = 'ok' AND cost_usd IS NOT NULL
        ORDER BY created_at DESC LIMIT $2
        """,
        model,
        sample,
    )
    if not rows:
        return None, 0
    total = sum((r["cost_usd"] for r in rows), Decimal(0))
    return total / len(rows), len(rows)
