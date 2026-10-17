"""SQL for the agent: pending actions and the tool-call audit. Same conventions as repository.py."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from backend.app.db import repository as repo

Db = repo.Db


class ActionRecord(BaseModel):
    id: UUID
    conversation_id: UUID
    workspace_id: UUID
    request_id: UUID
    tool: str
    args: dict[str, Any]
    risk: str
    summary: str
    status: str
    result: str | None
    created_at: datetime
    resolved_at: datetime | None


_COLUMNS = """id, conversation_id, workspace_id, request_id, tool, args, risk, summary, status,
              result, created_at, resolved_at"""


def _action(row: Any) -> ActionRecord:
    data = dict(row)
    if isinstance(data["args"], str):
        data["args"] = json.loads(data["args"])
    return ActionRecord(**data)


async def create_action(
    db: Db,
    *,
    conversation_id: UUID,
    workspace_id: UUID,
    request_id: UUID,
    tool: str,
    args: dict[str, Any],
    risk: str,
    summary: str,
) -> ActionRecord:
    row = await db.fetchrow(
        f"""
        INSERT INTO agent_actions (conversation_id, workspace_id, request_id, tool, args, risk, summary)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
        RETURNING {_COLUMNS}
        """,
        conversation_id,
        workspace_id,
        request_id,
        tool,
        json.dumps(args),
        risk,
        summary,
    )
    assert row is not None
    return _action(row)


async def get_action(db: Db, action_id: UUID) -> ActionRecord | None:
    row = await db.fetchrow(
        f"SELECT {_COLUMNS} FROM agent_actions WHERE id = $1", action_id
    )
    return _action(row) if row else None


async def list_actions(
    db: Db, conversation_id: UUID, *, pending_only: bool = False
) -> list[ActionRecord]:
    rows = await db.fetch(
        f"""
        SELECT {_COLUMNS} FROM agent_actions
        WHERE conversation_id = $1 AND ($2 = false OR status = 'pending')
        ORDER BY created_at DESC LIMIT 50
        """,
        conversation_id,
        pending_only,
    )
    return [_action(r) for r in rows]


async def claim_action(db: Db, action_id: UUID) -> ActionRecord | None:
    """Atomically take a pending action: of two simultaneous confirmations, exactly one wins, so a
    costly action can never run twice. Returns None if it was not pending."""
    row = await db.fetchrow(
        f"""
        UPDATE agent_actions SET status = 'confirmed', resolved_at = now()
        WHERE id = $1 AND status = 'pending'
        RETURNING {_COLUMNS}
        """,
        action_id,
    )
    return _action(row) if row else None


async def cancel_action(db: Db, action_id: UUID) -> ActionRecord | None:
    row = await db.fetchrow(
        f"""
        UPDATE agent_actions SET status = 'cancelled', resolved_at = now()
        WHERE id = $1 AND status = 'pending'
        RETURNING {_COLUMNS}
        """,
        action_id,
    )
    return _action(row) if row else None


async def finish_action(db: Db, action_id: UUID, *, ok: bool, result: str) -> None:
    await db.execute(
        "UPDATE agent_actions SET status = $2, result = $3 WHERE id = $1",
        action_id,
        "confirmed" if ok else "failed",
        result,
    )


async def record_tool_call(
    db: Db,
    *,
    conversation_id: UUID,
    request_id: UUID,
    step: int,
    tool: str,
    args: dict[str, Any],
    outcome: str,
    summary: str,
    latency_ms: int,
) -> None:
    await db.execute(
        """
        INSERT INTO tool_calls (conversation_id, request_id, step, tool, args, outcome, summary, latency_ms)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
        """,
        conversation_id,
        request_id,
        step,
        tool,
        json.dumps(args, default=str),
        outcome,
        summary[:500],
        latency_ms,
    )


async def conversation_spend(db: Db, conversation_id: UUID) -> tuple[Decimal, int]:
    """What the conversation's model calls have cost so far (only calls with a known cost), and how
    many calls had no known cost. Unknown calls are counted, never treated as free."""
    row = await db.fetchrow(
        """
        SELECT coalesce(sum(cost_usd), 0) AS spent,
               count(*) FILTER (WHERE cost_usd IS NULL AND cost_source = 'unknown') AS unknown
        FROM traces WHERE conversation_id = $1
        """,
        conversation_id,
    )
    assert row is not None
    return row["spent"], int(row["unknown"])
