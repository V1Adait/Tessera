"""All SQL for the app lives here. Every function takes a connection (or pool), so the CALLER
decides what runs in one transaction:

    async with pool.acquire() as conn, conn.transaction():
        msg = await add_message(conn, ...)
        await record_trace(conn, ...)

Values always go through $1, $2 placeholders, never into the SQL text.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg
from asyncpg.pool import PoolConnectionProxy
from pydantic import BaseModel

from backend.app.chunking.models import ChunkSet
from backend.app.db.records import (
    BlockRecord,
    ChunkHit,
    ChunkRecord,
    ConversationRecord,
    DocumentRecord,
    EntityRecord,
    MessageRole,
    MessageStatus,
    SectionRecord,
    StoredMessage,
    StoredSummary,
    TraceRecord,
    TrimEvent,
    WorkspaceRecord,
)
from backend.app.ingest.models import Block, assign_offsets

Db = asyncpg.Connection | asyncpg.Pool | PoolConnectionProxy

# --- conversations ----------------------------------------------------------------------


class DuplicateNameError(ValueError):
    """An entity with this name (ignoring case) already exists in the workspace."""


class WorkspaceNotFoundError(LookupError):
    """The workspace does not exist."""


# --- workspaces and entities ------------------------------------------------------------


async def create_workspace(db: Db, name: str) -> WorkspaceRecord:
    row = await db.fetchrow(
        "INSERT INTO workspaces (name) VALUES ($1) "
        "RETURNING id, name, created_at, updated_at",
        name.strip(),
    )
    assert row is not None
    return WorkspaceRecord(**dict(row))


async def list_workspaces(db: Db) -> list[WorkspaceRecord]:
    rows = await db.fetch(
        "SELECT id, name, created_at, updated_at FROM workspaces ORDER BY created_at, id"
    )
    return [WorkspaceRecord(**dict(r)) for r in rows]


async def get_workspace(db: Db, workspace_id: UUID) -> WorkspaceRecord | None:
    row = await db.fetchrow(
        "SELECT id, name, created_at, updated_at FROM workspaces WHERE id = $1",
        workspace_id,
    )
    return WorkspaceRecord(**dict(row)) if row else None


async def create_entity(db: Db, workspace_id: UUID, name: str) -> EntityRecord:
    """Add a row to the matrix. Raises WorkspaceNotFoundError or DuplicateNameError."""
    try:
        row = await db.fetchrow(
            """
            INSERT INTO entities (workspace_id, name) VALUES ($1, $2)
            RETURNING id, workspace_id, name, position, created_at
            """,
            workspace_id,
            name.strip(),
        )
    except asyncpg.UniqueViolationError as exc:
        raise DuplicateNameError(name) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise WorkspaceNotFoundError(str(workspace_id)) from exc
    assert row is not None
    return EntityRecord(**dict(row))


async def list_entities(db: Db, workspace_id: UUID) -> list[EntityRecord]:
    rows = await db.fetch(
        """
        SELECT id, workspace_id, name, position, created_at
        FROM entities WHERE workspace_id = $1 ORDER BY position
        """,
        workspace_id,
    )
    return [EntityRecord(**dict(r)) for r in rows]


# --- conversations ----------------------------------------------------------------------


async def create_conversation(
    db: Db, title: str | None = None, workspace_id: UUID | None = None
) -> UUID:
    """New conversation in `workspace_id`, or in the default (oldest) workspace when None.

    Raises WorkspaceNotFoundError for an unknown workspace.
    """
    try:
        row = await db.fetchrow(
            """
            INSERT INTO conversations (title, workspace_id)
            VALUES ($1, COALESCE($2::uuid, (SELECT id FROM workspaces ORDER BY created_at, id LIMIT 1)))
            RETURNING id
            """,
            title,
            workspace_id,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise WorkspaceNotFoundError(str(workspace_id)) from exc
    assert row is not None
    return row["id"]


async def get_conversation(db: Db, conversation_id: UUID) -> ConversationRecord | None:
    row = await db.fetchrow(
        """
        SELECT id, workspace_id, title, created_at, updated_at
        FROM conversations WHERE id = $1
        """,
        conversation_id,
    )
    return ConversationRecord(**dict(row)) if row else None


# --- messages ---------------------------------------------------------------------------


async def add_message(
    db: Db,
    conversation_id: UUID,
    role: MessageRole,
    content: str,
    *,
    message_id: UUID | None = None,
    status: MessageStatus = "complete",
    tokens: int | None = None,
    request_id: UUID | None = None,
) -> StoredMessage:
    """Insert a message and bump the conversation's updated_at in ONE statement.

    If the conversation does not exist the CTE returns no row, nothing is inserted, and we
    raise LookupError (instead of relying on a foreign-key error).
    """
    row = await db.fetchrow(
        """
        WITH touched AS (
            UPDATE conversations SET updated_at = now() WHERE id = $2 RETURNING id
        )
        INSERT INTO messages (id, conversation_id, role, content, status, tokens, request_id)
        SELECT COALESCE($1::uuid, gen_random_uuid()), touched.id, $3, $4, $5, $6, $7
        FROM touched
        RETURNING id, conversation_id, seq, role, content, status, tokens, request_id, created_at
        """,
        message_id,
        conversation_id,
        role,
        content,
        status,
        tokens,
        request_id,
    )
    if row is None:
        raise LookupError(f"conversation {conversation_id} does not exist")
    return StoredMessage(**dict(row))


async def finish_message(
    db: Db, message_id: UUID, content: str, status: MessageStatus, tokens: int | None
) -> None:
    """Complete a message that was inserted as 'streaming' (the assistant reply)."""
    result = await db.execute(
        "UPDATE messages SET content = $2, status = $3, tokens = $4 WHERE id = $1",
        message_id,
        content,
        status,
        tokens,
    )
    if result != "UPDATE 1":
        raise LookupError(f"message {message_id} does not exist")


async def list_messages(
    db: Db, conversation_id: UUID, *, after_message_id: UUID | None = None
) -> list[StoredMessage]:
    """Messages in order. With `after_message_id` (the summary pointer): only those after it."""
    after_seq = 0
    if after_message_id is not None:
        after_seq_row = await db.fetchrow(
            "SELECT seq FROM messages WHERE id = $1 AND conversation_id = $2",
            after_message_id,
            conversation_id,
        )
        if after_seq_row is None:
            raise ValueError(
                f"pointer {after_message_id} is not in conversation {conversation_id}"
            )
        after_seq = after_seq_row["seq"]

    rows = await db.fetch(
        """
        SELECT id, conversation_id, seq, role, content, status, tokens, request_id, created_at
        FROM messages
        WHERE conversation_id = $1 AND seq > $2
        ORDER BY seq
        """,
        conversation_id,
        after_seq,
    )
    return [StoredMessage(**dict(r)) for r in rows]


# --- summaries --------------------------------------------------------------------------


async def add_summary(
    db: Db,
    conversation_id: UUID,
    content: str,
    summarised_through_message_id: UUID,
    prompt_version: str,
    *,
    previous_summary_id: UUID | None = None,
    tokens: int | None = None,
    model: str | None = None,
    request_id: UUID | None = None,
) -> StoredSummary:
    """Store a summary round. The pointer must be a message of THIS conversation."""
    row = await db.fetchrow(
        """
        INSERT INTO summaries (conversation_id, previous_summary_id, summarised_through_message_id,
                               content, tokens, prompt_version, model, request_id)
        SELECT $1, $2, m.id, $4, $5, $6, $7, $8
        FROM messages m
        WHERE m.id = $3 AND m.conversation_id = $1
        RETURNING id, seq, conversation_id, previous_summary_id, summarised_through_message_id,
                  content, tokens, prompt_version, model, request_id, created_at
        """,
        conversation_id,
        previous_summary_id,
        summarised_through_message_id,
        content,
        tokens,
        prompt_version,
        model,
        request_id,
    )
    if row is None:
        raise ValueError(
            f"pointer {summarised_through_message_id} is not a message of conversation "
            f"{conversation_id}"
        )
    return StoredSummary(**dict(row))


async def latest_summary(db: Db, conversation_id: UUID) -> StoredSummary | None:
    row = await db.fetchrow(
        """
        SELECT id, seq, conversation_id, previous_summary_id, summarised_through_message_id,
               content, tokens, prompt_version, model, request_id, created_at
        FROM summaries WHERE conversation_id = $1 ORDER BY seq DESC LIMIT 1
        """,
        conversation_id,
    )
    return StoredSummary(**dict(row)) if row else None


async def load_history(
    db: Db, conversation_id: UUID
) -> tuple[StoredSummary | None, list[StoredMessage]]:
    """What the context builder needs: the active summary and the messages after its pointer."""
    summary = await latest_summary(db, conversation_id)
    pointer = summary.summarised_through_message_id if summary else None
    return summary, await list_messages(db, conversation_id, after_message_id=pointer)


# --- trim events ------------------------------------------------------------------------


async def record_trim_event(
    db: Db,
    conversation_id: UUID,
    request_id: UUID,
    *,
    messages_dropped: int,
    budget_tokens: int,
    tokens_dropped: int | None = None,
    dropped_through_message_id: UUID | None = None,
) -> TrimEvent:
    row = await db.fetchrow(
        """
        INSERT INTO trim_events (conversation_id, request_id, reason, messages_dropped,
                                 tokens_dropped, budget_tokens, dropped_through_message_id)
        VALUES ($1, $2, 'budget', $3, $4, $5, $6)
        RETURNING id, conversation_id, request_id, reason, messages_dropped, tokens_dropped,
                  budget_tokens, dropped_through_message_id, created_at
        """,
        conversation_id,
        request_id,
        messages_dropped,
        tokens_dropped,
        budget_tokens,
        dropped_through_message_id,
    )
    assert row is not None
    return TrimEvent(**dict(row))


async def list_trim_events(db: Db, conversation_id: UUID) -> list[TrimEvent]:
    rows = await db.fetch(
        """
        SELECT id, conversation_id, request_id, reason, messages_dropped, tokens_dropped,
               budget_tokens, dropped_through_message_id, created_at
        FROM trim_events WHERE conversation_id = $1 ORDER BY created_at, id
        """,
        conversation_id,
    )
    return [TrimEvent(**dict(r)) for r in rows]


# --- traces -----------------------------------------------------------------------------


async def record_trace(db: Db, trace: TraceRecord) -> None:
    """One row per request. A duplicate request_id raises UniqueViolationError (a bug)."""
    await db.execute(
        """
        INSERT INTO traces (request_id, conversation_id, kind, provider, model, prompt_version,
                            tokens_in, tokens_out, cost_usd, cost_source, latency_ms,
                            first_token_ms, status, error_code)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """,
        trace.request_id,
        trace.conversation_id,
        trace.kind,
        trace.provider,
        trace.model,
        trace.prompt_version,
        trace.tokens_in,
        trace.tokens_out,
        trace.cost_usd,
        trace.cost_source,
        trace.latency_ms,
        trace.first_token_ms,
        trace.status,
        trace.error_code,
    )


async def list_traces(
    db: Db, *, conversation_id: UUID | None = None, limit: int = 50
) -> list[TraceRecord]:
    """Newest first. Optionally only the traces of one conversation."""
    rows = await db.fetch(
        """
        SELECT request_id, conversation_id, kind, provider, model, prompt_version, tokens_in,
               tokens_out, cost_usd, cost_source, latency_ms, first_token_ms, status, error_code,
               created_at
        FROM traces
        WHERE ($1::uuid IS NULL OR conversation_id = $1)
        ORDER BY created_at DESC, request_id
        LIMIT $2
        """,
        conversation_id,
        limit,
    )
    return [TraceRecord(**dict(r)) for r in rows]


# --- documents and their parsed blocks --------------------------------------------------


class EntityNotInWorkspaceError(LookupError):
    """The entity does not exist, or belongs to a different workspace than the document."""


class DuplicateDocumentError(Exception):
    """This workspace already has a document with the same bytes. `existing` is that document."""

    def __init__(self, existing: DocumentRecord) -> None:
        super().__init__(f"duplicate of document {existing.id}")
        self.existing = existing


def _json(value: object) -> dict[str, Any] | None:
    """asyncpg hands jsonb back as text unless a codec is set; accept both."""
    if value is None:
        return None
    parsed: Any = json.loads(value) if isinstance(value, str) else value
    return dict(parsed)


def _document(row: asyncpg.Record) -> DocumentRecord:
    data = dict(row)
    data["report"] = _json(data["report"]) or {}
    return DocumentRecord(**data)


def _block(row: asyncpg.Record) -> BlockRecord:
    data = dict(row)
    data["location"] = _json(data["location"]) or {}
    data["data"] = _json(data["data"])
    return BlockRecord(**data)


async def default_workspace_id(db: Db) -> UUID:
    row = await db.fetchrow("SELECT id FROM workspaces ORDER BY created_at, id LIMIT 1")
    if row is None:
        raise WorkspaceNotFoundError("no workspace exists")
    return row["id"]


async def entity_in_workspace(db: Db, entity_id: UUID, workspace_id: UUID) -> bool:
    row = await db.fetchrow(
        "SELECT 1 FROM entities WHERE id = $1 AND workspace_id = $2",
        entity_id,
        workspace_id,
    )
    return row is not None


async def create_document(
    db: Db,
    *,
    workspace_id: UUID,
    entity_id: UUID | None,
    filename: str,
    kind: str,
    sha256: str,
    size_bytes: int,
) -> DocumentRecord:
    """Record an uploaded file (status 'uploaded').

    Raises WorkspaceNotFoundError, EntityNotInWorkspaceError, or DuplicateDocumentError (the
    same bytes are already a document in this workspace; the error carries that document)."""
    if await get_workspace(db, workspace_id) is None:
        raise WorkspaceNotFoundError(str(workspace_id))
    if entity_id is not None and not await entity_in_workspace(
        db, entity_id, workspace_id
    ):
        raise EntityNotInWorkspaceError(str(entity_id))
    try:
        row = await db.fetchrow(
            """
            INSERT INTO documents (workspace_id, entity_id, filename, kind, sha256, size_bytes)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, workspace_id, entity_id, filename, kind, sha256, size_bytes, status,
                      status_detail, parser, page_count, char_count, report, created_at, parsed_at,
               index_status, index_detail, indexed_model, indexed_at
            """,
            workspace_id,
            entity_id,
            filename,
            kind,
            sha256,
            size_bytes,
        )
    except asyncpg.UniqueViolationError as exc:
        existing = await db.fetchrow(
            """
            SELECT id, workspace_id, entity_id, filename, kind, sha256, size_bytes, status,
                   status_detail, parser, page_count, char_count, report, created_at, parsed_at,
               index_status, index_detail, indexed_model, indexed_at
            FROM documents WHERE workspace_id = $1 AND sha256 = $2
            """,
            workspace_id,
            sha256,
        )
        if existing is None:  # deleted in the meantime: treat it as a lost race
            raise
        raise DuplicateDocumentError(_document(existing)) from exc
    assert row is not None
    return _document(row)


async def get_document(db: Db, document_id: UUID) -> DocumentRecord | None:
    row = await db.fetchrow(
        """
        SELECT id, workspace_id, entity_id, filename, kind, sha256, size_bytes, status,
               status_detail, parser, page_count, char_count, report, created_at, parsed_at,
               index_status, index_detail, indexed_model, indexed_at
        FROM documents WHERE id = $1
        """,
        document_id,
    )
    return _document(row) if row else None


async def list_documents(
    db: Db,
    workspace_id: UUID,
    *,
    entity_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRecord]:
    rows = await db.fetch(
        """
        SELECT id, workspace_id, entity_id, filename, kind, sha256, size_bytes, status,
               status_detail, parser, page_count, char_count, report, created_at, parsed_at,
               index_status, index_detail, indexed_model, indexed_at
        FROM documents
        WHERE workspace_id = $1 AND ($2::uuid IS NULL OR entity_id = $2)
        ORDER BY created_at DESC, id
        LIMIT $3 OFFSET $4
        """,
        workspace_id,
        entity_id,
        limit,
        offset,
    )
    return [_document(r) for r in rows]


async def assign_entity(db: Db, document_id: UUID, entity_id: UUID | None) -> None:
    """Attach a document to an entity of the same workspace (None un-assigns it)."""
    try:
        result = await db.execute(
            "UPDATE documents SET entity_id = $2 WHERE id = $1", document_id, entity_id
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise EntityNotInWorkspaceError(str(entity_id)) from exc
    if result != "UPDATE 1":
        raise LookupError(f"document {document_id} does not exist")


async def claim_for_parsing(db: Db, document_id: UUID) -> DocumentRecord | None:
    """Mark a document 'parsing' unless a worker already has it. Returns None if it is already
    being parsed (or does not exist), so two workers can never parse the same document."""
    row = await db.fetchrow(
        """
        UPDATE documents SET status = 'parsing', status_detail = NULL
        WHERE id = $1 AND status <> 'parsing'
        RETURNING id, workspace_id, entity_id, filename, kind, sha256, size_bytes, status,
                  status_detail, parser, page_count, char_count, report, created_at, parsed_at,
               index_status, index_detail, indexed_model, indexed_at
        """,
        document_id,
    )
    return _document(row) if row else None


async def set_document_status(
    db: Db, document_id: UUID, status: str, detail: str | None = None
) -> None:
    await db.execute(
        "UPDATE documents SET status = $2, status_detail = $3 WHERE id = $1",
        document_id,
        status,
        detail,
    )


async def recover_interrupted(db: Db) -> int:
    """After a restart, documents left 'uploaded' or 'parsing' will never finish by themselves.
    Mark them failed (with a reason) so they show up and can be re-parsed."""
    result = await db.execute(
        """
        UPDATE documents
        SET status = 'failed', status_detail = 'interrupted by a restart: use re-parse'
        WHERE status IN ('uploaded', 'parsing')
        """
    )
    return int(result.split()[-1])


def _block_rows(document_id: UUID, blocks: list[Block]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for seq, (block, (start, end)) in enumerate(
        zip(blocks, assign_offsets(blocks), strict=True)
    ):
        if block.table is not None:
            data: dict[str, Any] | None = block.table.model_dump()
        elif block.labels:
            data = {"labels": block.labels}
        else:
            data = None
        rows.append(
            (
                document_id,
                seq,
                block.type,
                block.text,
                block.level,
                json.dumps(block.location.model_dump(mode="json", exclude_none=True)),
                json.dumps(data) if data is not None else None,
                block.confidence,
                block.flags,
                start,
                end,
            )
        )
    return rows


async def save_parse_result(
    pool: asyncpg.Pool,
    document_id: UUID,
    *,
    parser: str,
    status: str,
    detail: str | None,
    page_count: int | None,
    char_count: int,
    report: dict[str, Any],
    blocks: list[Block],
    chunks: ChunkSet | None = None,
) -> None:
    """Replace a document's blocks (and its chunks) and set its status, in ONE transaction: a
    document never has half its blocks, or chunks that belong to older blocks, and a re-parse
    swaps the old ones for the new ones atomically."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "DELETE FROM document_blocks WHERE document_id = $1", document_id
        )
        await conn.executemany(
            """
            INSERT INTO document_blocks (document_id, seq, type, text, level, location, data,
                                         confidence, flags, char_start, char_end)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10, $11)
            """,
            _block_rows(document_id, blocks),
        )
        await replace_chunks(conn, document_id, chunks or ChunkSet())
        result = await conn.execute(
            """
            UPDATE documents
            SET status = $2, status_detail = $3, parser = $4, page_count = $5,
                char_count = $6, report = $7::jsonb, parsed_at = now(),
                index_status = 'pending', index_detail = NULL
            WHERE id = $1
            """,
            document_id,
            status,
            detail,
            parser,
            page_count,
            char_count,
            json.dumps(report),
        )
        if result != "UPDATE 1":
            raise LookupError(f"document {document_id} does not exist")


async def list_blocks(
    db: Db,
    document_id: UUID,
    *,
    block_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[BlockRecord]:
    rows = await db.fetch(
        """
        SELECT id, document_id, seq, type, text, level, location, data, confidence, flags,
               char_start, char_end
        FROM document_blocks
        WHERE document_id = $1 AND ($2::text IS NULL OR type = $2)
        ORDER BY seq
        LIMIT $3 OFFSET $4
        """,
        document_id,
        block_type,
        limit,
        offset,
    )
    return [_block(r) for r in rows]


async def count_blocks(db: Db, document_id: UUID, block_type: str | None = None) -> int:
    count = await db.fetchval(
        """
        SELECT count(*) FROM document_blocks
        WHERE document_id = $1 AND ($2::text IS NULL OR type = $2)
        """,
        document_id,
        block_type,
    )
    return int(count)


# --- chunks -----------------------------------------------------------------------------


def _chunk(row: asyncpg.Record) -> ChunkRecord:
    data = {k: v for k, v in dict(row).items() if k in ChunkRecord.model_fields}
    return ChunkRecord(**data)


_CHUNK_COLUMNS = """
    c.id, c.document_id, c.section_id, c.ordinal, c.kind, c.text, c.search_text,
    c.section_path, c.block_seqs, c.char_start, c.char_end, c.token_count, c.page_start,
    c.page_end, c.sheet, c.slide, c.cell_range, c.units, c.confidence, c.flags, c.content_hash
"""


async def replace_chunks(
    db: asyncpg.Connection | PoolConnectionProxy, document_id: UUID, chunk_set: ChunkSet
) -> None:
    """Swap a document's sections and chunks for a new set. Call it inside a transaction (it
    takes a connection, not a pool): the old chunks are gone only if the new ones are in."""
    await db.execute(
        "DELETE FROM document_sections WHERE document_id = $1", document_id
    )
    section_ids: dict[int, UUID] = {}
    for parent in chunk_set.parents:
        section_ids[parent.ordinal] = await db.fetchval(
            """
            INSERT INTO document_sections (document_id, ordinal, path, char_start, char_end,
                                           page_start, page_end, token_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            document_id,
            parent.ordinal,
            parent.path,
            parent.char_start,
            parent.char_end,
            parent.page_start,
            parent.page_end,
            parent.token_count,
        )
    await db.executemany(
        """
        INSERT INTO document_chunks (document_id, section_id, ordinal, kind, text, search_text,
                                     section_path, block_seqs, char_start, char_end, token_count,
                                     page_start, page_end, sheet, slide, cell_range, units,
                                     confidence, flags, content_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
                $18, $19, $20)
        """,
        [
            (
                document_id,
                section_ids[c.parent_ordinal],
                c.ordinal,
                c.kind,
                c.text,
                c.search_text,
                c.section_path,
                c.block_seqs,
                c.char_start,
                c.char_end,
                c.token_count,
                c.page_start,
                c.page_end,
                c.sheet,
                c.slide,
                c.cell_range,
                c.units,
                c.confidence,
                c.flags,
                c.content_hash,
            )
            for c in chunk_set.chunks
        ],
    )


async def save_chunks(
    pool: asyncpg.Pool, document_id: UUID, chunk_set: ChunkSet
) -> None:
    """Rebuild a document's chunks without touching its blocks (for example after it was
    assigned to an entity, which changes the search text). One transaction."""
    async with pool.acquire() as conn, conn.transaction():
        await replace_chunks(conn, document_id, chunk_set)
        await conn.execute(
            "UPDATE documents SET index_status = 'pending', index_detail = NULL WHERE id = $1",
            document_id,
        )


async def list_chunks(
    db: Db, document_id: UUID, *, limit: int = 100, offset: int = 0
) -> list[ChunkRecord]:
    rows = await db.fetch(
        f"""
        SELECT {_CHUNK_COLUMNS} FROM document_chunks c
        WHERE c.document_id = $1
        ORDER BY c.ordinal LIMIT $2 OFFSET $3
        """,
        document_id,
        limit,
        offset,
    )
    return [_chunk(r) for r in rows]


async def count_chunks(db: Db, document_id: UUID) -> int:
    count = await db.fetchval(
        "SELECT count(*) FROM document_chunks WHERE document_id = $1", document_id
    )
    return int(count)


async def get_chunk(db: Db, chunk_id: UUID) -> ChunkRecord | None:
    row = await db.fetchrow(
        f"SELECT {_CHUNK_COLUMNS} FROM document_chunks c WHERE c.id = $1", chunk_id
    )
    return _chunk(row) if row else None


async def list_sections(db: Db, document_id: UUID) -> list[SectionRecord]:
    rows = await db.fetch(
        """
        SELECT id, document_id, ordinal, path, char_start, char_end, page_start, page_end,
               token_count
        FROM document_sections WHERE document_id = $1 ORDER BY ordinal
        """,
        document_id,
    )
    return [SectionRecord(**dict(r)) for r in rows]


async def get_section_chunks(db: Db, section_id: UUID) -> list[ChunkRecord]:
    """Every chunk of one parent, in reading order: what the model reads for a found chunk."""
    rows = await db.fetch(
        f"""
        SELECT {_CHUNK_COLUMNS} FROM document_chunks c
        WHERE c.section_id = $1 ORDER BY c.ordinal
        """,
        section_id,
    )
    return [_chunk(r) for r in rows]


async def entity_name(db: Db, entity_id: UUID | None) -> str | None:
    if entity_id is None:
        return None
    name: str | None = await db.fetchval(
        "SELECT name FROM entities WHERE id = $1", entity_id
    )
    return name


async def search_chunks_text(
    db: Db,
    workspace_id: UUID,
    query: str,
    *,
    entity_ids: list[UUID] | None = None,
    limit: int = 10,
) -> list[ChunkHit]:
    """Full-text search over search_text, inside one workspace (and optionally some entities).

    The scope filter runs in SQL BEFORE ranking, so another workspace's text can never appear.
    `websearch_to_tsquery` reads plain words safely (quotes, OR, -word) and never raises on odd
    input. Ranking is ts_rank_cd: a first baseline to be measured, not a final choice."""
    rows = await db.fetch(
        f"""
        SELECT {_CHUNK_COLUMNS}, d.filename,
               ts_rank_cd(c.search_tsv, websearch_to_tsquery('english', $2)) AS score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.workspace_id = $1
          AND ($3::uuid[] IS NULL OR d.entity_id = ANY($3))
          AND c.search_tsv @@ websearch_to_tsquery('english', $2)
        ORDER BY score DESC, c.document_id, c.ordinal
        LIMIT $4
        """,
        workspace_id,
        query,
        entity_ids,
        limit,
    )
    return [
        ChunkHit(chunk=_chunk(r), filename=r["filename"], score=float(r["score"]))
        for r in rows
    ]


async def get_sections(db: Db, section_ids: list[UUID]) -> list[SectionRecord]:
    rows = await db.fetch(
        """
        SELECT id, document_id, ordinal, path, char_start, char_end, page_start, page_end,
               token_count
        FROM document_sections WHERE id = ANY($1::uuid[])
        """,
        section_ids,
    )
    return [SectionRecord(**dict(r)) for r in rows]


# --- the vector index: per-document state ----------------------------------------------------


async def claim_for_indexing(db: Db, document_id: UUID) -> bool:
    """pending -> indexing, atomically. True means THIS worker owns the job; False means someone
    else has it, it is already indexed, or the document is not ready (not parsed yet)."""
    result = await db.execute(
        """
        UPDATE documents SET index_status = 'indexing', index_detail = NULL
        WHERE id = $1 AND index_status = 'pending' AND status IN ('parsed', 'needs_review')
        """,
        document_id,
    )
    return result == "UPDATE 1"


async def set_index_result(
    db: Db,
    document_id: UUID,
    status: str,
    detail: str | None = None,
    model: str | None = None,
) -> bool:
    """Record how indexing ended. Only applies while the document is still 'indexing': if it was
    re-parsed meanwhile (back to 'pending'), that newer state wins and False is returned."""
    result = await db.execute(
        """
        UPDATE documents
        SET index_status = $2, index_detail = $3,
            indexed_model = CASE WHEN $2 = 'indexed' THEN $4 ELSE indexed_model END,
            indexed_at = CASE WHEN $2 = 'indexed' THEN now() ELSE indexed_at END
        WHERE id = $1 AND index_status = 'indexing'
        """,
        document_id,
        status,
        detail,
        model,
    )
    return result == "UPDATE 1"


async def release_indexing(db: Db, document_id: UUID) -> None:
    """Put a document we were indexing back to 'pending' (shutdown, cancellation)."""
    await db.execute(
        "UPDATE documents SET index_status = 'pending' WHERE id = $1 AND index_status = 'indexing'",
        document_id,
    )


async def recover_interrupted_indexing(db: Db) -> int:
    """At startup: documents left 'indexing' by a crash go back to 'pending'."""
    result = await db.execute(
        "UPDATE documents SET index_status = 'pending' WHERE index_status = 'indexing'"
    )
    return int(result.split()[-1])


async def list_documents_to_index(db: Db, limit: int = 1000) -> list[UUID]:
    rows = await db.fetch(
        """
        SELECT id FROM documents
        WHERE index_status = 'pending' AND status IN ('parsed', 'needs_review')
        ORDER BY created_at LIMIT $1
        """,
        limit,
    )
    return [r["id"] for r in rows]


async def all_chunks(db: Db, document_id: UUID) -> list[ChunkRecord]:
    """Every chunk of a document, in order (paged so one query never grows without bound)."""
    found: list[ChunkRecord] = []
    while True:
        page = await list_chunks(db, document_id, limit=500, offset=len(found))
        found += page
        if len(page) < 500:
            return found


async def chunks_in_workspace(
    db: Db, workspace_id: UUID, chunk_ids: list[UUID]
) -> dict[UUID, tuple[ChunkRecord, str]]:
    """These chunks with their document's file name, but only those whose document is in this
    workspace. A vector hit is never trusted on its own: Postgres has the last word on scope."""
    rows = await db.fetch(
        f"""
        SELECT {_CHUNK_COLUMNS}, d.filename
        FROM document_chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY($1::uuid[]) AND d.workspace_id = $2
        """,
        chunk_ids,
        workspace_id,
    )
    return {r["id"]: (_chunk(r), r["filename"]) for r in rows}


async def document_scope(db: Db, document_id: UUID) -> tuple[UUID, UUID | None] | None:
    """(workspace_id, entity_id) of a document, or None if it does not exist."""
    row = await db.fetchrow(
        "SELECT workspace_id, entity_id FROM documents WHERE id = $1", document_id
    )
    return (row["workspace_id"], row["entity_id"]) if row else None


async def mark_pending_index(db: Db, document_id: UUID) -> bool:
    """Ask for (re)indexing: any state except 'indexing' goes back to 'pending'. False if the
    document does not exist or is being indexed right now."""
    result = await db.execute(
        """
        UPDATE documents SET index_status = 'pending', index_detail = NULL
        WHERE id = $1 AND index_status <> 'indexing'
        """,
        document_id,
    )
    return result == "UPDATE 1"


class CostLine(BaseModel):
    """Spend for one kind of call and model over a period."""

    kind: str
    model: str | None
    calls: int
    unknown_cost_calls: (
        int  # paid calls whose cost could not be worked out: NOT counted as free
    )
    estimated_calls: int  # calls whose cost is computed from prices, not reported
    cost_usd: Decimal
    tokens_in: int
    tokens_out: int


async def cost_summary(db: Db, since: datetime) -> list[CostLine]:
    """What was spent since `since`, by kind of call and model, biggest first."""
    rows = await db.fetch(
        """
        SELECT kind, model, count(*) AS calls,
               count(*) FILTER (WHERE cost_source = 'unknown') AS unknown_cost_calls,
               count(*) FILTER (WHERE cost_source = 'computed') AS estimated_calls,
               coalesce(sum(cost_usd), 0) AS cost_usd,
               coalesce(sum(tokens_in), 0)::bigint AS tokens_in,
               coalesce(sum(tokens_out), 0)::bigint AS tokens_out
        FROM traces WHERE created_at >= $1
        GROUP BY kind, model
        ORDER BY cost_usd DESC, kind, model
        """,
        since,
    )
    return [CostLine(**dict(r)) for r in rows]
