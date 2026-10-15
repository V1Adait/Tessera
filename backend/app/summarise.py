"""Rolling conversation summaries, written by a cheap model AFTER a turn finished.

    prompt grew past ~70% of its budget
        -> plan_summary: everything before the last few turns is folded into a summary
        -> the cheap model writes the new summary from (previous summary + those turns)
        -> checks: not empty, shorter than what it replaces
        -> ONE transaction: store the summary (with its pointer) + a trace row

Rules this file keeps:
  * Nothing here may break or slow a chat: it runs in the background and never raises.
  * Nothing is deleted. A summary only moves a pointer; the full conversation stays stored.
  * A failed or rejected summary changes nothing: the old summary stays, and a later turn retries.
  * It is an LLM call, so it gets its own trace row (kind='summary') like any other call.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

import asyncpg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.app.config import Settings
from backend.app.context import (
    estimate_tokens,
    input_budget,
    message_tokens,
    needs_summary,
    plan_summary,
)
from backend.app.costs import Cost, get_price_book, reported_cost, resolve_cost
from backend.app.db import repository as repo
from backend.app.db.records import StoredSummary, TraceRecord
from backend.app.llm import classify_error
from backend.app.models import Message
from backend.app.prompts import SUMMARY_PROMPT, SUMMARY_PROMPT_VERSION

logger = logging.getLogger(__name__)

SummaryOutcome = Literal[
    "stored", "not_needed", "disabled", "busy", "failed", "rejected", "stale"
]

# Conversations being summarised right now in this process: stops two turns from starting two
# summaries of the same conversation. (A database lock re-checks this at store time.)
_in_progress: set[UUID] = set()

_TAG_NAMES = "conversation|previous_summary"
_TAG_RE = re.compile(rf"<(/?)({_TAG_NAMES})", re.IGNORECASE)


def _neutralise(text: str) -> str:
    """Stop conversation text from closing our <conversation> block and escaping into the
    instructions: '</conversation>' inside a message becomes '</_conversation>'."""
    return _TAG_RE.sub(r"<\1_\2", text)


def build_summary_prompt(
    previous_summary: str | None, messages: list[Message]
) -> list[BaseMessage]:
    """System instructions in one slot; the conversation to fold in is DATA in another."""
    transcript = "\n".join(f"[{m.role}] {_neutralise(m.content)}" for m in messages)
    previous = _neutralise(previous_summary) if previous_summary else "(none yet)"
    return [
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(
            content=(
                f"<previous_summary>\n{previous}\n</previous_summary>\n\n"
                f"<conversation>\n{transcript}\n</conversation>\n\n"
                "Write the updated summary now."
            )
        ),
    ]


def cap_to_budget(messages: list[Message], budget_tokens: int) -> list[Message]:
    """The oldest messages that fit in `budget_tokens`, ending on an assistant message so the
    pointer lands on a turn boundary. A huge backlog is therefore summarised in several rounds."""
    kept: list[Message] = []
    used = 0
    for message in messages:
        cost = message_tokens(message)
        if used + cost > budget_tokens:
            break
        kept.append(message)
        used += cost
    while kept and kept[-1].role != "assistant":
        kept.pop()
    return kept


@dataclass(frozen=True)
class _Generated:
    text: str
    tokens_in: int | None
    tokens_out: int | None
    cost: Cost


async def _generate(
    llm: BaseChatModel,
    previous: str | None,
    messages: list[Message],
    *,
    provider: str,
    model: str,
) -> _Generated:
    reply = await llm.ainvoke(build_summary_prompt(previous, messages))
    usage = reply.usage_metadata  # type: ignore[attr-defined]  # present on AIMessage
    tokens_in = usage["input_tokens"] if usage else None
    tokens_out = usage["output_tokens"] if usage else None
    return _Generated(
        text=str(reply.text).strip(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=resolve_cost(
            reported=reported_cost(getattr(reply, "response_metadata", None)),
            book=get_price_book(),
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        ),
    )


def _check(text: str, folded: list[Message]) -> str | None:
    """Reasons to refuse a summary (None means it is fine)."""
    if not text:
        return "empty_summary"
    replaced = sum(len(m.content) for m in folded)
    if len(text) >= replaced:
        return "summary_not_shorter"  # it would cost more context than it saves
    return None


async def maybe_summarise(
    pool: asyncpg.Pool,
    settings: Settings,
    llm: BaseChatModel | None,
    conversation_id: UUID,
    *,
    tokens_used: int,
) -> SummaryOutcome:
    """Summarise the conversation if its prompt has used enough of the budget. Never raises."""
    if llm is None:
        return "disabled"
    if not needs_summary(tokens_used, input_budget(settings), settings.summarize_at):
        return "not_needed"
    if conversation_id in _in_progress:
        return "busy"

    _in_progress.add(conversation_id)
    try:
        return await _summarise(pool, settings, llm, conversation_id)
    except Exception:
        logger.exception(
            "summarising failed", extra={"conversation_id": str(conversation_id)}
        )
        return "failed"
    finally:
        _in_progress.discard(conversation_id)


async def _summarise(
    pool: asyncpg.Pool, settings: Settings, llm: BaseChatModel, conversation_id: UUID
) -> SummaryOutcome:
    previous, stored = await repo.load_history(pool, conversation_id)
    history = [
        Message(id=m.id, role=m.role, content=m.content)
        for m in stored
        if m.usable_in_prompt
    ]
    plan = plan_summary(history, settings.recent_turns)
    folded = cap_to_budget(
        plan.to_summarise,
        settings.summary_input_max_tokens
        - estimate_tokens(previous.content if previous else "")
        - estimate_tokens(SUMMARY_PROMPT),
    )
    if not folded:
        return "not_needed"  # fewer than recent_turns turns, or nothing fits: nothing to fold

    request_id = uuid4()
    started = time.perf_counter()
    base_trace = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "kind": "summary",
        "provider": settings.summary_provider,
        "model": settings.summary_model,
        "prompt_version": SUMMARY_PROMPT_VERSION,
    }

    try:
        generated = await _generate(
            llm,
            previous.content if previous else None,
            folded,
            provider=settings.summary_provider,
            model=settings.summary_model,
        )
    except Exception as exc:  # noqa: BLE001 - any provider failure becomes a trace row
        code, _, _ = classify_error(exc)
        logger.warning(
            "summary call failed: %s",
            code,
            extra={"conversation_id": str(conversation_id)},
        )
        await _trace(
            pool,
            TraceRecord(
                **base_trace,  # type: ignore[arg-type]
                status="error",
                error_code=code,
                latency_ms=_ms_since(started),
            ),
        )
        return "failed"

    latency_ms = _ms_since(started)
    problem = _check(generated.text, folded)
    if problem is not None:
        logger.warning("summary refused: %s", problem)
        await _trace(
            pool,
            TraceRecord(
                **base_trace,  # type: ignore[arg-type]
                status="rejected",
                error_code=problem,
                tokens_in=generated.tokens_in,
                tokens_out=generated.tokens_out,
                cost_usd=generated.cost.usd,
                cost_source=generated.cost.source,
                latency_ms=latency_ms,
            ),
        )
        return "rejected"

    return await _store(
        pool,
        conversation_id,
        previous,
        folded[-1].id,
        generated,
        base_trace,
        latency_ms,
        settings,
    )


async def _store(
    pool: asyncpg.Pool,
    conversation_id: UUID,
    previous: StoredSummary | None,
    pointer: UUID,
    generated: _Generated,
    base_trace: dict[str, object],
    latency_ms: int,
    settings: Settings,
) -> SummaryOutcome:
    """Store the summary and its trace in one transaction, unless another summary got there first."""
    try:
        async with pool.acquire() as conn, conn.transaction():
            # Serialise writers for this conversation, then re-check that nothing moved while
            # the model was thinking (two turns finishing close together).
            await conn.execute(
                "SELECT 1 FROM conversations WHERE id = $1 FOR UPDATE", conversation_id
            )
            latest = await repo.latest_summary(conn, conversation_id)
            if (latest.id if latest else None) != (previous.id if previous else None):
                logger.info(
                    "discarding a stale summary",
                    extra={"conversation_id": str(conversation_id)},
                )
                return "stale"
            await repo.add_summary(
                conn,
                conversation_id,
                generated.text,
                pointer,
                SUMMARY_PROMPT_VERSION,
                previous_summary_id=previous.id if previous else None,
                tokens=generated.tokens_out,
                model=settings.summary_model,
                request_id=base_trace["request_id"],  # type: ignore[arg-type]
            )
            await repo.record_trace(
                conn,
                TraceRecord(
                    **base_trace,  # type: ignore[arg-type]
                    status="ok",
                    tokens_in=generated.tokens_in,
                    tokens_out=generated.tokens_out,
                    cost_usd=generated.cost.usd,
                    cost_source=generated.cost.source,
                    latency_ms=latency_ms,
                ),
            )
    except Exception:
        logger.exception(
            "failed to store the summary",
            extra={"conversation_id": str(conversation_id)},
        )
        return "failed"
    return "stored"


async def _trace(pool: asyncpg.Pool, trace: TraceRecord) -> None:
    try:
        await repo.record_trace(pool, trace)
    except Exception:
        logger.exception("failed to record the summary trace")


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
