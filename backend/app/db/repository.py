"""All SQL for the app lives here. Every function takes a connection (or pool), so the CALLER
decides what runs in one transaction:

    async with pool.acquire() as conn, conn.transaction():
        msg = await add_message(conn, ...)
        await record_trace(conn, ...)

Values always go through $1, $2 placeholders, never into the SQL text.
"""

from uuid import UUID

import asyncpg
from asyncpg.pool import PoolConnectionProxy

from backend.app.db.records import (
    ConversationRecord,
    EntityRecord,
    MessageRole,
    MessageStatus,
    StoredMessage,
    StoredSummary,
    TraceRecord,
    TrimEvent,
    WorkspaceRecord,
)

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
                            tokens_in, tokens_out, cost_usd, latency_ms, first_token_ms,
                            status, error_code)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
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
               tokens_out, cost_usd, latency_ms, first_token_ms, status, error_code, created_at
        FROM traces
        WHERE ($1::uuid IS NULL OR conversation_id = $1)
        ORDER BY created_at DESC, request_id
        LIMIT $2
        """,
        conversation_id,
        limit,
    )
    return [TraceRecord(**dict(r)) for r in rows]
