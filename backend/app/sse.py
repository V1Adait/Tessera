"""Turning a chat turn's events into Server-Sent Events. Shared by every route that streams an answer
(/chat, and the answer written from filled cells), so they cannot drift apart."""

import logging
from collections.abc import AsyncIterator
from uuid import UUID

import asyncpg
from langchain_core.language_models import BaseChatModel

from backend.app import chat_service
from backend.app.agent.runtime import AgentRuntime
from backend.app.config import Settings
from backend.app.models import ErrorEvent, StreamEvent

logger = logging.getLogger(__name__)


def sse(event: StreamEvent) -> str:
    """One Server-Sent Event: 'data: <json>' followed by a blank line."""
    return f"data: {event.model_dump_json()}\n\n"


def stream_headers(request_id: UUID, conversation_id: UUID) -> dict[str, str]:
    return {
        "X-Request-ID": str(request_id),
        "X-Conversation-ID": str(conversation_id),
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # stops nginx-style proxies from buffering the stream
    }


async def sse_events(
    pool: asyncpg.Pool,
    settings: Settings,
    llm: BaseChatModel,
    turn: chat_service.Turn,
    summary_llm: BaseChatModel | None,
    *,
    agent: AgentRuntime | None = None,
    system_extra: str | None = None,
) -> AsyncIterator[str]:
    """The turn's events as SSE lines. A crash becomes one ErrorEvent, never a broken stream."""
    try:
        async for event in chat_service.stream_turn(
            pool, settings, llm, turn, summary_llm, agent, system_extra=system_extra
        ):
            yield sse(event)
    except Exception:
        logger.exception("turn crashed", extra={"request_id": str(turn.request_id)})
        yield sse(
            ErrorEvent(
                request_id=turn.request_id,
                code="unknown",
                message="Unexpected error while handling the request.",
                retryable=False,
            )
        )
