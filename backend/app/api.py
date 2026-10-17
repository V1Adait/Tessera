"""HTTP layer: turns requests into chat turns and turn events into SSE lines.

No business logic here. Everything the routes need (settings, model, database pool) arrives
through Depends, so tests can replace them with fakes.
"""

import logging
from typing import Annotated
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from backend.app import chat_service
from backend.app.api_agent import AgentDep
from backend.app.config import Settings, get_settings
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import StoredMessage
from backend.app.llm import get_default_llm, get_summary_llm
from backend.app.models import ChatInput
from backend.app.sse import sse, sse_events, stream_headers

logger = logging.getLogger(__name__)

# `sse` stays importable from here: the helper moved to sse.py, and tests and callers use api.sse.
__all__ = ["router", "sse"]

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(get_settings)]
LlmDep = Annotated[BaseChatModel, Depends(get_default_llm)]
PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]
SummaryLlmDep = Annotated[BaseChatModel | None, Depends(get_summary_llm)]


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
    agent: AgentDep,
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

    return StreamingResponse(
        sse_events(pool, settings, llm, turn, summary_llm, agent=agent),
        media_type="text/event-stream",
        headers=stream_headers(request_id, turn.conversation_id),
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
