"""HTTP layer: turns requests into chat turns and turn events into SSE lines.

No business logic here. Everything the routes need (settings, model, database pool) arrives
through Depends, so tests can replace them with fakes.
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from backend.app import chat_service
from backend.app.config import Settings, get_settings
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import StoredMessage
from backend.app.llm import get_default_llm, get_summary_llm
from backend.app.models import ChatInput, ErrorEvent, StreamEvent

logger = logging.getLogger(__name__)

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(get_settings)]
LlmDep = Annotated[BaseChatModel, Depends(get_default_llm)]
PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]
SummaryLlmDep = Annotated[BaseChatModel | None, Depends(get_summary_llm)]


def sse(event: StreamEvent) -> str:
    """One Server-Sent Event: 'data: <json>' followed by a blank line."""
    return f"data: {event.model_dump_json()}\n\n"


class ConversationView(BaseModel):
    id: UUID
    workspace_id: UUID
    title: str | None
    messages: list[StoredMessage]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat")
async def chat(
    body: ChatInput,
    settings: SettingsDep,
    llm: LlmDep,
    pool: PoolDep,
    summary_llm: SummaryLlmDep,
) -> StreamingResponse:
    """Send ONE message; stream the model's answer as SSE (Delta events, then Done or Error).

    The conversation id comes back in the X-Conversation-ID header (a new conversation is
    created when the body has none). Problems found BEFORE streaming get a real status code:
    422 for a malformed body, 404 for an unknown conversation. Once streaming starts the
    status is already 200, so later failures arrive as an ErrorEvent inside the stream.
    """
    request_id = uuid4()
    try:
        turn = await chat_service.begin_turn(
            pool, body.conversation_id, body.message, request_id, body.workspace_id
        )
    except chat_service.ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversation not found") from None
    except chat_service.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None

    async def events() -> AsyncIterator[str]:
        try:
            async for event in chat_service.stream_turn(
                pool, settings, llm, turn, summary_llm
            ):
                yield sse(event)
        except Exception:
            logger.exception("turn crashed", extra={"request_id": str(request_id)})
            yield sse(
                ErrorEvent(
                    request_id=request_id,
                    code="unknown",
                    message="Unexpected error while handling the request.",
                    retryable=False,
                )
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": str(request_id),
            "X-Conversation-ID": str(turn.conversation_id),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx-style proxies from buffering the stream
        },
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationView)
async def get_conversation(conversation_id: UUID, pool: PoolDep) -> ConversationView:
    """The full stored conversation (every message, including ones a summary covers)."""
    conversation = await repo.get_conversation(pool, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = await repo.list_messages(pool, conversation_id)
    return ConversationView(
        id=conversation.id,
        workspace_id=conversation.workspace_id,
        title=conversation.title,
        messages=messages,
    )
