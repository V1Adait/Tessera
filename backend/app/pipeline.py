"""The pipeline: runs one chat request from start to finish.

    request -> input checks -> build_context -> stream_chat -> output checks -> events

The route (api.py) only handles HTTP; the pieces (context.py, llm.py) only do their own job.
This file is the one place that wires them together, so it is also where guardrails,
tracing and error handling attach later. It knows nothing about HTTP.
"""

import logging
from collections.abc import AsyncIterator, Callable
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from backend.app.config import Settings
from backend.app.context import BuiltContext, ContextTooLargeError, assemble_context
from backend.app.llm import stream_chat
from backend.app.models import ChatRequest, ErrorEvent, StreamEvent

logger = logging.getLogger(__name__)


def check_input(request: ChatRequest, settings: Settings) -> None:
    """Input guardrails slot. No-op for now.

    Later: raise an error when a check fails (empty message, too long, rate limited, ...).
    """


def check_output(event: StreamEvent) -> StreamEvent:
    """Output guardrails slot. No-op for now: returns the event unchanged.

    Later: validate citations, moderate text, and so on.
    """
    return event


async def run(
    request: ChatRequest,
    settings: Settings,
    llm: BaseChatModel,
    request_id: UUID | None = None,
    on_context: Callable[[BuiltContext], None] | None = None,
    summary: str | None = None,
    system_extra: str | None = None,
    generate: Callable[[list[BaseMessage]], AsyncIterator[StreamEvent]] | None = None,
) -> AsyncIterator[StreamEvent]:
    """ChatRequest in, a stream of Delta events then one Done (or one ErrorEvent) out.

    The llm and settings come in as arguments, not fetched here, so tests can pass fakes.
    `on_context` is told what was sent (and what was trimmed) once the prompt is built, so a
    caller can record it without the pipeline knowing about databases.
    `generate`, when given, replaces the plain model call: the agent uses it to run its tool loop
    on the same finished prompt. `system_extra` adds rules to the system prompt (the agent's).
    Failures end the stream with an ErrorEvent; they never raise to the caller.
    """
    request_id = request_id or uuid4()

    check_input(request, settings)

    try:
        built = assemble_context(
            request, settings, summary=summary, system_extra=system_extra
        )
    except ContextTooLargeError:
        logger.warning("context too large", extra={"request_id": str(request_id)})
        yield ErrorEvent(
            request_id=request_id,
            code="context_too_large",
            message="The message is too large for the model's context window. Shorten it.",
            retryable=False,
        )
        return
    except ValueError:
        # build_context rejects a conversation that does not end with a user message.
        logger.warning("invalid conversation", extra={"request_id": str(request_id)})
        yield ErrorEvent(
            request_id=request_id,
            code="bad_request",
            message="The conversation must end with a user message.",
            retryable=False,
        )
        return

    if on_context is not None:
        on_context(built)

    events = (
        generate(built.messages)
        if generate is not None
        else stream_chat(built.messages, llm, request_id)
    )
    async for event in events:
        yield check_output(event)
