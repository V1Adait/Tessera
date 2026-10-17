"""HTTP routes for the agent's pending actions: see what is waiting, confirm it, or cancel it.

    GET  /conversations/{id}/actions      what is still waiting for a yes
    POST /actions/{id}/confirm            run it (once, however many times this is sent)
    POST /actions/{id}/cancel             drop it

Thin handlers: the rules are in agent/actions.py.
"""

from typing import Annotated
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel

from backend.app.agent.actions import (
    ActionExpiredError,
    ActionNotFoundError,
    ActionNotPendingError,
    ActionOutcome,
)
from backend.app.agent.runtime import AgentRuntime
from backend.app.answer.prompts import ANSWER_RULES
from backend.app.answer.service import ActionMissingError, NotReadyError, prepare_answer
from backend.app.config import Settings, get_settings
from backend.app.db.agent_repository import ActionRecord
from backend.app.db.pool import get_pool
from backend.app.llm import get_default_llm, get_summary_llm
from backend.app.sse import sse_events, stream_headers

router = APIRouter(tags=["agent"])


def get_agent(request: Request) -> AgentRuntime | None:
    """The agent, or None when it is off or not configured: chat then answers without tools."""
    return getattr(request.app.state, "agent", None)  # type: ignore[no-any-return]


AgentDep = Annotated[AgentRuntime | None, Depends(get_agent)]


def _required(agent: AgentRuntime | None) -> AgentRuntime:
    if agent is None:
        raise HTTPException(status_code=503, detail="the agent is not configured")
    return agent


@router.get(
    "/conversations/{conversation_id}/actions", response_model=list[ActionRecord]
)
async def pending_actions(conversation_id: UUID, agent: AgentDep) -> list[ActionRecord]:
    return await _required(agent).desk.pending(conversation_id)


@router.post("/actions/{action_id}/confirm", response_model=ActionOutcome)
async def confirm_action(action_id: UUID, agent: AgentDep) -> ActionOutcome:
    try:
        return await _required(agent).desk.confirm(action_id)
    except ActionNotFoundError:
        raise HTTPException(status_code=404, detail="action not found") from None
    except ActionNotPendingError:
        raise HTTPException(
            status_code=409, detail="this action was already confirmed or cancelled"
        ) from None
    except ActionExpiredError:
        raise HTTPException(
            status_code=410,
            detail="this action is too old: the matrix may have changed. Ask again.",
        ) from None


@router.post("/actions/{action_id}/cancel", response_model=ActionOutcome)
async def cancel_action(action_id: UUID, agent: AgentDep) -> ActionOutcome:
    try:
        return await _required(agent).desk.cancel(action_id)
    except ActionNotFoundError:
        raise HTTPException(status_code=404, detail="action not found") from None
    except ActionNotPendingError:
        raise HTTPException(
            status_code=409, detail="this action was already confirmed or cancelled"
        ) from None


@router.post("/actions/{action_id}/answer")
async def answer_from_cells(
    action_id: UUID,
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[BaseChatModel, Depends(get_default_llm)],
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
    summary_llm: Annotated[BaseChatModel | None, Depends(get_summary_llm)],
) -> StreamingResponse:
    """Write the answer to a confirmed plan's question from the cells it filled, as an SSE stream
    like /chat. 409 while the table is still being filled, or if the plan was not confirmed."""
    request_id = uuid4()
    try:
        turn = await prepare_answer(pool, action_id, request_id)
    except ActionMissingError:
        raise HTTPException(status_code=404, detail="plan not found") from None
    except NotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return StreamingResponse(
        sse_events(pool, settings, llm, turn, summary_llm, system_extra=ANSWER_RULES),
        media_type="text/event-stream",
        headers=stream_headers(request_id, turn.conversation_id),
    )
