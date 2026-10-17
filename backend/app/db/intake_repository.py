"""SQL for intake results: what was decided for each file. Same conventions as repository.py."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from backend.app.db import repository as repo

Db = repo.Db


class IntakeRecord(BaseModel):
    document_id: UUID
    outcome: str
    entity_id: UUID | None
    detected: dict[str, Any]
    detail: str | None
    candidates: list[str]
    created_at: datetime


_COLUMNS = "document_id, outcome, entity_id, detected, detail, candidates, created_at"


def _record(row: Any) -> IntakeRecord:
    data = dict(row)
    if isinstance(data["detected"], str):
        data["detected"] = json.loads(data["detected"])
    return IntakeRecord(**data)


async def save_result(
    db: Db,
    document_id: UUID,
    *,
    outcome: str,
    entity_id: UUID | None,
    detected: dict[str, Any],
    detail: str | None,
    candidates: list[str],
) -> IntakeRecord:
    row = await db.fetchrow(
        f"""
        INSERT INTO intake_results (document_id, outcome, entity_id, detected, detail, candidates)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        ON CONFLICT (document_id) DO UPDATE
        SET outcome = EXCLUDED.outcome, entity_id = EXCLUDED.entity_id,
            detected = EXCLUDED.detected, detail = EXCLUDED.detail,
            candidates = EXCLUDED.candidates, created_at = now()
        RETURNING {_COLUMNS}
        """,
        document_id,
        outcome,
        entity_id,
        json.dumps(detected),
        detail,
        candidates,
    )
    assert row is not None
    return _record(row)


async def get_result(db: Db, document_id: UUID) -> IntakeRecord | None:
    row = await db.fetchrow(
        f"SELECT {_COLUMNS} FROM intake_results WHERE document_id = $1", document_id
    )
    return _record(row) if row else None
