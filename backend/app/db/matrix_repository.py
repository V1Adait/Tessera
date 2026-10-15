"""SQL for the matrix: columns (with versions), cells and their evidence.

Same conventions as repository.py: every function takes a connection or pool, values go through
$1, $2 placeholders, and the caller decides what shares a transaction. Functions that must
change several tables at once open their own transaction, so a column is never half-updated
and a cell is never saved without its evidence.
"""

import json
from typing import Any
from uuid import UUID

import asyncpg

from backend.app.db import repository as repo
from backend.app.matrix.models import (
    CellOutcome,
    CellRecord,
    ColumnDraft,
    ColumnPatch,
    ColumnRecord,
    EvidenceRecord,
)

Db = repo.Db


class ColumnNotFoundError(LookupError):
    """No such column in this workspace."""


class DuplicateColumnError(ValueError):
    """A column with this name (ignoring case) already exists in the workspace."""


class CellPinnedError(Exception):
    """A person set this cell. Replacing it needs an explicit yes."""


class MissingEvidenceError(ValueError):
    """An answered cell was about to be saved without the evidence its column requires."""


def _name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def _column(row: asyncpg.Record) -> ColumnRecord:
    return ColumnRecord(**dict(row))


_COLUMN_SELECT = """
    SELECT c.id, c.workspace_id, c.position, c.current_version AS version,
           v.name, v.prompt, v.type, v.allowed_values, v.applicability, v.evidence_required,
           v.depends_on, c.created_at, c.updated_at
    FROM matrix_columns c
    JOIN column_versions v ON v.column_id = c.id AND v.version = c.current_version
"""


# --- columns ----------------------------------------------------------------------------------


async def _check_depends_on(db: Db, workspace_id: UUID, depends_on: list[UUID]) -> None:
    if not depends_on:
        return
    found = await db.fetchval(
        "SELECT count(*) FROM matrix_columns WHERE workspace_id = $1 AND id = ANY($2::uuid[])",
        workspace_id,
        depends_on,
    )
    if found != len(set(depends_on)):
        raise ColumnNotFoundError("a column this one depends on does not exist here")


async def _insert_version(
    conn: asyncpg.Connection | Any, column_id: UUID, version: int, draft: ColumnDraft
) -> None:
    await conn.execute(
        """
        INSERT INTO column_versions (column_id, version, name, prompt, type, allowed_values,
                                     applicability, evidence_required, depends_on)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        column_id,
        version,
        draft.name,
        draft.prompt,
        draft.type,
        draft.allowed_values,
        draft.applicability,
        draft.evidence_required,
        draft.depends_on,
    )


async def create_column(
    pool: asyncpg.Pool, workspace_id: UUID, draft: ColumnDraft
) -> ColumnRecord:
    """Raises WorkspaceNotFoundError, DuplicateColumnError or ColumnNotFoundError."""
    async with pool.acquire() as conn, conn.transaction():
        if await repo.get_workspace(conn, workspace_id) is None:
            raise repo.WorkspaceNotFoundError(str(workspace_id))
        await _check_depends_on(conn, workspace_id, draft.depends_on)
        try:
            column_id = await conn.fetchval(
                """
                INSERT INTO matrix_columns (workspace_id, name_key) VALUES ($1, $2) RETURNING id
                """,
                workspace_id,
                _name_key(draft.name),
            )
        except asyncpg.UniqueViolationError:
            raise DuplicateColumnError(draft.name) from None
        await _insert_version(conn, column_id, 1, draft)
        row = await conn.fetchrow(_COLUMN_SELECT + " WHERE c.id = $1", column_id)
    assert row is not None
    return _column(row)


async def get_column(db: Db, column_id: UUID) -> ColumnRecord | None:
    row = await db.fetchrow(_COLUMN_SELECT + " WHERE c.id = $1", column_id)
    return _column(row) if row else None


async def list_columns(db: Db, workspace_id: UUID) -> list[ColumnRecord]:
    rows = await db.fetch(
        _COLUMN_SELECT + " WHERE c.workspace_id = $1 ORDER BY c.position", workspace_id
    )
    return [_column(r) for r in rows]


def apply_patch(current: ColumnRecord, patch: ColumnPatch) -> ColumnDraft:
    """The draft a column becomes when `patch` is applied. Validation (a select needs values, and
    so on) happens when the draft is built, so a bad patch changes nothing."""
    changes = patch.model_dump(exclude_unset=True, exclude={"clear_applicability"})
    merged = {
        "name": current.name,
        "prompt": current.prompt,
        "type": current.type,
        "allowed_values": current.allowed_values,
        "applicability": current.applicability,
        "evidence_required": current.evidence_required,
        "depends_on": current.depends_on,
    }
    for key, value in changes.items():
        if value is not None:
            merged[key] = value
    if patch.clear_applicability:
        merged["applicability"] = None
    if (
        patch.type is not None
        and patch.type not in ("select", "status")
        and patch.allowed_values is None
    ):
        merged["allowed_values"] = []  # moving away from a select drops its list
    return ColumnDraft(**merged)  # type: ignore[arg-type]


async def update_column(
    pool: asyncpg.Pool, column_id: UUID, patch: ColumnPatch
) -> ColumnRecord:
    """Change a column. A real change makes a new version and marks the cells that answered the
    old one stale; a patch that changes nothing makes no version. Raises ColumnNotFoundError,
    DuplicateColumnError, or ValueError (a draft that is not valid)."""
    async with pool.acquire() as conn, conn.transaction():
        locked = await conn.fetchrow(
            "SELECT workspace_id FROM matrix_columns WHERE id = $1 FOR UPDATE",
            column_id,
        )
        if locked is None:
            raise ColumnNotFoundError(str(column_id))
        row = await conn.fetchrow(_COLUMN_SELECT + " WHERE c.id = $1", column_id)
        assert row is not None
        current = _column(row)
        draft = apply_patch(current, patch)
        unchanged = draft == ColumnDraft(
            name=current.name,
            prompt=current.prompt,
            type=current.type,
            allowed_values=current.allowed_values,
            applicability=current.applicability,
            evidence_required=current.evidence_required,
            depends_on=current.depends_on,
        )
        if unchanged:
            return current
        await _check_depends_on(conn, current.workspace_id, draft.depends_on)
        if column_id in draft.depends_on:
            raise ValueError("a column cannot depend on itself")
        version = current.version + 1
        try:
            await conn.execute(
                """
                UPDATE matrix_columns
                SET current_version = $2, name_key = $3, updated_at = now() WHERE id = $1
                """,
                column_id,
                version,
                _name_key(draft.name),
            )
        except asyncpg.UniqueViolationError:
            raise DuplicateColumnError(draft.name) from None
        await _insert_version(conn, column_id, version, draft)
        await conn.execute(
            "UPDATE cells SET stale = true WHERE column_id = $1", column_id
        )
        row = await conn.fetchrow(_COLUMN_SELECT + " WHERE c.id = $1", column_id)
    assert row is not None
    return _column(row)


# --- cells ------------------------------------------------------------------------------------

_CELL_COLUMNS = """
    id, workspace_id, entity_id, column_id, column_version, status, status_detail, value,
    display_value, as_of, derivation, stale, pinned, model, prompt_version, created_at, updated_at
"""


def _json_value(value: Any) -> Any:
    """asyncpg returns jsonb as text unless a codec is set: accept both."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _evidence(row: asyncpg.Record) -> EvidenceRecord:
    return EvidenceRecord(**dict(row))


async def _attach(db: Db, rows: list[asyncpg.Record]) -> list[CellRecord]:
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    evidence_rows = await db.fetch(
        """
        SELECT id, cell_id, seq, document_id, chunk_id, page_start, page_end, char_start,
               char_end, exact_range, quote
        FROM cell_evidence WHERE cell_id = ANY($1::uuid[]) ORDER BY cell_id, seq
        """,
        ids,
    )
    by_cell: dict[UUID, list[EvidenceRecord]] = {}
    for e in evidence_rows:
        by_cell.setdefault(e["cell_id"], []).append(_evidence(e))
    cells: list[CellRecord] = []
    for r in rows:
        data = dict(r)
        data["value"] = _json_value(data["value"])
        data["derivation"] = _json_value(data["derivation"])
        cells.append(CellRecord(**data, evidence=by_cell.get(r["id"], [])))
    return cells


async def get_cell(db: Db, cell_id: UUID) -> CellRecord | None:
    row = await db.fetchrow(f"SELECT {_CELL_COLUMNS} FROM cells WHERE id = $1", cell_id)
    return (await _attach(db, [row]))[0] if row else None


async def get_cell_for(db: Db, entity_id: UUID, column_id: UUID) -> CellRecord | None:
    row = await db.fetchrow(
        f"SELECT {_CELL_COLUMNS} FROM cells WHERE entity_id = $1 AND column_id = $2",
        entity_id,
        column_id,
    )
    return (await _attach(db, [row]))[0] if row else None


async def list_cells(
    db: Db,
    workspace_id: UUID,
    *,
    entity_id: UUID | None = None,
    column_id: UUID | None = None,
) -> list[CellRecord]:
    rows = await db.fetch(
        f"""
        SELECT {_CELL_COLUMNS} FROM cells
        WHERE workspace_id = $1 AND ($2::uuid IS NULL OR entity_id = $2)
          AND ($3::uuid IS NULL OR column_id = $3)
        ORDER BY column_id, entity_id
        """,
        workspace_id,
        entity_id,
        column_id,
    )
    return await _attach(db, rows)


async def save_cell(
    pool: asyncpg.Pool,
    *,
    workspace_id: UUID,
    entity_id: UUID,
    column: ColumnRecord,
    outcome: CellOutcome,
    replace_pinned: bool = False,
) -> CellRecord:
    """Store the outcome of one extraction, replacing any earlier cell for the same row and column.

    One transaction: the cell and its evidence are written together or not at all. Raises
    CellPinnedError (a person set the existing cell and `replace_pinned` is false),
    MissingEvidenceError (an answered cell, but its column requires evidence and there is none),
    or the database's foreign-key error for an entity outside the workspace."""
    if (
        outcome.status == "answered"
        and column.evidence_required
        and not outcome.evidence
    ):
        raise MissingEvidenceError(
            f"answered cell for column {column.name!r} has no evidence"
        )
    async with pool.acquire() as conn, conn.transaction():
        existing = await conn.fetchrow(
            "SELECT id, pinned FROM cells WHERE entity_id = $1 AND column_id = $2 FOR UPDATE",
            entity_id,
            column.id,
        )
        if existing is not None and existing["pinned"] and not replace_pinned:
            raise CellPinnedError(str(existing["id"]))
        value = json.dumps(outcome.value) if outcome.value is not None else None
        if existing is None:
            cell_id = await conn.fetchval(
                """
                INSERT INTO cells (workspace_id, entity_id, column_id, column_version, status,
                                   status_detail, value, display_value, model, prompt_version)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10) RETURNING id
                """,
                workspace_id,
                entity_id,
                column.id,
                column.version,
                outcome.status,
                outcome.detail,
                value,
                outcome.display,
                outcome.model,
                outcome.prompt_version,
            )
        else:
            cell_id = existing["id"]
            await conn.execute("DELETE FROM cell_evidence WHERE cell_id = $1", cell_id)
            await conn.execute(
                """
                UPDATE cells SET column_version = $2, status = $3, status_detail = $4,
                       value = $5::jsonb, display_value = $6, model = $7, prompt_version = $8,
                       stale = false, pinned = false, updated_at = now()
                WHERE id = $1
                """,
                cell_id,
                column.version,
                outcome.status,
                outcome.detail,
                value,
                outcome.display,
                outcome.model,
                outcome.prompt_version,
            )
        await conn.executemany(
            """
            INSERT INTO cell_evidence (cell_id, seq, document_id, chunk_id, page_start, page_end,
                                       char_start, char_end, exact_range, quote)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            [
                (
                    cell_id,
                    seq,
                    e.document_id,
                    e.chunk_id,
                    e.page_start,
                    e.page_end,
                    e.char_start,
                    e.char_end,
                    e.exact_range,
                    e.quote,
                )
                for seq, e in enumerate(outcome.evidence)
            ],
        )
        cell = await get_cell(conn, cell_id)
    assert cell is not None
    return cell


async def set_pinned(db: Db, cell_id: UUID, pinned: bool) -> bool:
    """Mark a cell as set by a person (or release it). False if there is no such cell."""
    result = await db.execute(
        "UPDATE cells SET pinned = $2 WHERE id = $1", cell_id, pinned
    )
    return result == "UPDATE 1"
