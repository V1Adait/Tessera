"""Persistence around one chat turn. The server owns the conversation.

    begin_turn   (before streaming)  store the user message, load history + summary
    stream_turn  (while streaming)   run the pipeline, pass events through, then ALWAYS record
                                     the assistant message, the trace and any trim event

Recording happens in a `finally`, so it also runs when the client disconnects (Stop button,
closed tab). A failure to record is logged and swallowed: the database must never break the chat.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
from langchain_core.language_models import BaseChatModel

from backend.app import pipeline
from backend.app.config import Settings
from backend.app.context import BuiltContext, estimate_tokens
from backend.app.costs import Cost, get_price_book, resolve_cost
from backend.app.db import repository as repo
from backend.app.db.records import (
    MessageStatus,
    TraceRecord,
    TraceStatus,
)
from backend.app.models import (
    ChatRequest,
    Delta,
    Done,
    ErrorEvent,
    Message,
    StreamEvent,
)
from backend.app.prompts import PROMPT_VERSION
from backend.app.summarise import maybe_summarise

logger = logging.getLogger(__name__)

# Recording tasks that must outlive a cancelled request. Holding a reference stops the event
# loop from garbage-collecting a task that is still running.
_background: set[asyncio.Task[Any]] = set()


class ConversationNotFoundError(LookupError):
    """The client named a conversation that does not exist."""


class WorkspaceNotFoundError(LookupError):
    """The client named a workspace that does not exist."""


@dataclass(frozen=True)
class Turn:
    """Everything stream_turn needs, prepared by begin_turn."""

    conversation_id: UUID
    request_id: UUID
    request: (
        ChatRequest  # the unsummarised history from the database + the new user message
    )
    summary: str | None  # the active summary text, if any


async def begin_turn(
    pool: asyncpg.Pool,
    conversation_id: UUID | None,
    text: str,
    request_id: UUID,
    workspace_id: UUID | None = None,
) -> Turn:
    """Store the user's message (creating the conversation if needed) and load the history.

    `workspace_id` only matters for a NEW conversation (None means the default workspace)."""
    async with pool.acquire() as conn, conn.transaction():
        if conversation_id is None:
            try:
                conversation_id = await repo.create_conversation(
                    conn, title=text.strip()[:60], workspace_id=workspace_id
                )
            except repo.WorkspaceNotFoundError as exc:
                raise WorkspaceNotFoundError(str(workspace_id)) from exc
        try:
            await repo.add_message(
                conn, conversation_id, "user", text, request_id=request_id
            )
        except LookupError as exc:
            raise ConversationNotFoundError(str(conversation_id)) from exc
        summary, stored = await repo.load_history(conn, conversation_id)

    messages = [
        Message(id=m.id, role=m.role, content=m.content)
        for m in stored
        if m.usable_in_prompt
    ]
    return Turn(
        conversation_id=conversation_id,
        request_id=request_id,
        request=ChatRequest(messages=messages),
        summary=summary.content if summary else None,
    )


async def stream_turn(
    pool: asyncpg.Pool,
    settings: Settings,
    llm: BaseChatModel,
    turn: Turn,
    summary_llm: BaseChatModel | None = None,
) -> AsyncIterator[StreamEvent]:
    started = time.perf_counter()
    first_token_ms: int | None = None
    parts: list[str] = []
    terminal: Done | ErrorEvent | None = None
    built: BuiltContext | None = None
    crashed = False

    def on_context(context: BuiltContext) -> None:
        nonlocal built
        built = context

    try:
        async for event in pipeline.run(
            turn.request,
            settings,
            llm,
            turn.request_id,
            on_context=on_context,
            summary=turn.summary,
        ):
            if isinstance(event, Delta):
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000)
                parts.append(event.text)
            elif isinstance(event, (Done, ErrorEvent)):
                terminal = event
            yield event
    except Exception:
        crashed = True
        raise
    finally:
        # Runs on a normal end, an error, a client disconnect and aclose().
        outcome = _Outcome(
            text="".join(parts),
            terminal=terminal,
            built=built,
            crashed=crashed,
            latency_ms=round((time.perf_counter() - started) * 1000),
            first_token_ms=first_token_ms,
        )
        task = _spawn(_after_turn(pool, settings, turn, outcome, summary_llm))
        # Shield: if this request is being cancelled, the recording keeps running anyway.
        await asyncio.shield(task)


def _spawn(coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Run a coroutine in the background and keep a reference so it is not garbage-collected."""
    task = asyncio.ensure_future(coroutine)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task


async def _after_turn(
    pool: asyncpg.Pool,
    settings: Settings,
    turn: Turn,
    outcome: _Outcome,
    summary_llm: BaseChatModel | None,
) -> None:
    """Record the turn first (that is what the request waits for), then maybe summarise."""
    await _record(pool, settings, turn, outcome)

    done = outcome.terminal if isinstance(outcome.terminal, Done) else None
    if done is None or summary_llm is None:
        return
    # The provider's own count of what this turn sent is exact; fall back to our estimate.
    tokens_used = done.usage.tokens_in or (
        estimate_tokens(" ".join(str(m.content) for m in outcome.built.messages))
        if outcome.built
        else 0
    )
    # A separate task: the request must not wait for a slow summary call.
    _spawn(
        maybe_summarise(
            pool, settings, summary_llm, turn.conversation_id, tokens_used=tokens_used
        )
    )


@dataclass(frozen=True)
class _Outcome:
    text: str
    terminal: Done | ErrorEvent | None
    built: BuiltContext | None
    crashed: bool
    latency_ms: int
    first_token_ms: int | None


def _turn_cost(settings: Settings, done: Done | None, text: str) -> Cost | None:
    """What a chat turn cost. Streamed responses carry no provider-reported cost, so a finished turn
    is priced from its token counts (labelled 'computed'). A turn that stopped part-way but had
    already produced text was still billed by the provider for it, so it is 'unknown', never free.
    A turn that never reached the model costs nothing and has no cost at all."""
    if done is not None:
        return resolve_cost(
            reported=None,
            book=get_price_book(),
            provider=settings.provider,
            model=settings.default_model,
            tokens_in=done.usage.tokens_in,
            tokens_out=done.usage.tokens_out,
        )
    if text:
        return Cost(None, "unknown")
    return None


def _classify(
    outcome: _Outcome,
) -> tuple[TraceStatus, MessageStatus | None, str | None]:
    """(trace status, assistant message status or None for no message, error code)."""
    if outcome.crashed:
        return "error", "error", "unknown"
    terminal = outcome.terminal
    if terminal is None:
        return "cancelled", "stopped", None  # client went away or pressed Stop
    if isinstance(terminal, Done):
        return "ok", "complete", None
    if outcome.built is None:
        return "rejected", None, terminal.code  # refused before the model was called
    return "error", "error", terminal.code


async def _record(
    pool: asyncpg.Pool, settings: Settings, turn: Turn, outcome: _Outcome
) -> None:
    """Write the assistant message, trim event and trace in ONE transaction. Never raises."""
    trace_status, message_status, error_code = _classify(outcome)
    done = outcome.terminal if isinstance(outcome.terminal, Done) else None
    cost = _turn_cost(settings, done, outcome.text)
    try:
        async with pool.acquire() as conn, conn.transaction():
            if message_status is not None:
                await repo.add_message(
                    conn,
                    turn.conversation_id,
                    "assistant",
                    outcome.text,
                    status=message_status,
                    tokens=done.usage.tokens_out if done else None,
                    request_id=turn.request_id,
                )
            if outcome.built is not None and outcome.built.trim is not None:
                trim = outcome.built.trim
                await repo.record_trim_event(
                    conn,
                    turn.conversation_id,
                    turn.request_id,
                    messages_dropped=trim.messages_dropped,
                    budget_tokens=trim.budget_tokens,
                    tokens_dropped=trim.tokens_dropped,
                    dropped_through_message_id=trim.dropped_through_message_id,
                )
            await repo.record_trace(
                conn,
                TraceRecord(
                    request_id=turn.request_id,
                    conversation_id=turn.conversation_id,
                    kind="chat",
                    status=trace_status,
                    provider=settings.provider,
                    model=settings.default_model,
                    prompt_version=PROMPT_VERSION,
                    tokens_in=done.usage.tokens_in if done else None,
                    tokens_out=done.usage.tokens_out if done else None,
                    cost_usd=cost.usd if cost else None,
                    cost_source=cost.source if cost else None,
                    latency_ms=outcome.latency_ms,
                    first_token_ms=outcome.first_token_ms,
                    error_code=error_code,
                ),
            )
    except Exception:
        logger.exception(
            "failed to record the turn", extra={"request_id": str(turn.request_id)}
        )
