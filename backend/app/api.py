"""HTTP layer: turns requests into pipeline calls and pipeline events into SSE lines.

No business logic here. Everything the route needs (settings, model) arrives through
Depends, so tests can replace them with fakes.
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel

from backend.app import pipeline
from backend.app.config import Settings, get_settings
from backend.app.llm import get_default_llm
from backend.app.models import ChatRequest, ErrorEvent, StreamEvent

logger = logging.getLogger(__name__)

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(get_settings)]
LlmDep = Annotated[BaseChatModel, Depends(get_default_llm)]


def sse(event: StreamEvent) -> str:
    """One Server-Sent Event: 'data: <json>' followed by a blank line."""
    return f"data: {event.model_dump_json()}\n\n"


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat")
async def chat(
    request: ChatRequest, settings: SettingsDep, llm: LlmDep
) -> StreamingResponse:
    """Stream the model's answer as SSE: Delta events, then one Done or one ErrorEvent.

    Once streaming starts the HTTP status is already 200, so failures are reported as an
    ErrorEvent inside the stream, not as a status code. Only a malformed request body is
    rejected up front (422) by FastAPI.
    """
    request_id = uuid4()

    async def events() -> AsyncIterator[str]:
        try:
            async for event in pipeline.run(request, settings, llm, request_id):
                yield sse(event)
        except Exception:
            logger.exception("pipeline crashed", extra={"request_id": str(request_id)})
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx-style proxies from buffering the stream
        },
    )
