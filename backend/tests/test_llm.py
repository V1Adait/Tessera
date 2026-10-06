import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import anthropic
import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from backend.app.llm import stream_chat, to_lc_messages
from backend.app.models import ChatRequest, Delta, Done, ErrorEvent, Message


def make_request(**kwargs) -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="hi")], **kwargs)


class ProviderLikeFake(BaseChatModel):
    """Streams like a real provider: empty leading chunks, text, then usage on an empty chunk."""

    @property
    def _llm_type(self) -> str:
        return "provider-like-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        for text in ["", "", "hello", " there"]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        usage = UsageMetadata(input_tokens=5, output_tokens=3, total_tokens=8)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=usage)
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="")
        )  # trailing chunk, no usage


def fake_llm(message: AIMessage) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([message]))


def test_to_lc_messages_with_system_prompt():
    req = ChatRequest(
        system="be brief",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "again"},
        ],
    )
    msgs = to_lc_messages(req)

    assert [type(m) for m in msgs] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [m.content for m in msgs] == ["be brief", "hi", "yo", "again"]


def test_to_lc_messages_without_system_prompt():
    msgs = to_lc_messages(make_request())

    assert [type(m) for m in msgs] == [HumanMessage]


def test_to_lc_messages_maps_system_role_in_list():
    req = ChatRequest(
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hi"},
        ]
    )

    assert [type(m) for m in to_lc_messages(req)] == [SystemMessage, HumanMessage]


async def test_stream_chat_yields_deltas_then_done():
    events = [e async for e in stream_chat(make_request(), ProviderLikeFake())]

    assert isinstance(events[-1], Done)
    assert all(isinstance(e, Delta) for e in events[:-1])
    assert "".join(e.text for e in events[:-1]) == "hello there"


async def test_stream_chat_reports_usage_from_late_empty_chunk():
    events = [e async for e in stream_chat(make_request(), ProviderLikeFake())]
    done = events[-1]

    assert isinstance(done, Done)
    assert done.usage.tokens_in == 5
    assert done.usage.tokens_out == 3


async def test_stream_chat_skips_empty_chunks():
    events = [e async for e in stream_chat(make_request(), ProviderLikeFake())]

    assert all(e.text for e in events if isinstance(e, Delta))


async def test_stream_chat_missing_usage_warns_and_reports_zeros(caplog):
    llm = fake_llm(AIMessage(content="hello"))

    with caplog.at_level(logging.WARNING):
        events = [e async for e in stream_chat(make_request(), llm)]
    done = events[-1]

    assert isinstance(done, Done)
    assert (done.usage.tokens_in, done.usage.tokens_out) == (0, 0)
    assert "no usage metadata" in caplog.text


# --- errors and request id ---------------------------------------------------


# The SDKs type `request` against their own vendored httpx; a plain httpx.Request works at runtime.
_REQUEST: Any = httpx.Request("POST", "https://example.invalid")


def _status_error(cls, status: int):
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status, request=request)
    return cls("boom with secret-ish detail", response=response, body=None)


class FailingFake(BaseChatModel):
    """Raises `exc` when streamed. Set `after_text` to fail mid-stream."""

    exc: Exception
    after_text: str = ""

    @property
    def _llm_type(self) -> str:
        return "failing-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.after_text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.after_text))
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "code", "retryable"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "auth", False),
        (_status_error(openai.AuthenticationError, 401), "auth", False),
        (_status_error(openai.NotFoundError, 404), "model_not_found", False),
        (_status_error(anthropic.RateLimitError, 429), "rate_limited", True),
        (_status_error(openai.InternalServerError, 500), "unavailable", True),
        (_status_error(openai.BadRequestError, 400), "bad_request", False),
        (
            openai.APITimeoutError(request=_REQUEST),
            "timeout",
            True,
        ),
        (
            anthropic.APIConnectionError(request=_REQUEST),
            "unavailable",
            True,
        ),
        (RuntimeError("something else"), "unknown", False),
    ],
)
async def test_provider_errors_become_error_events(exc, code, retryable):
    events = [e async for e in stream_chat(make_request(), FailingFake(exc=exc))]

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ErrorEvent)
    assert event.code == code
    assert event.retryable is retryable
    assert (
        "secret-ish" not in event.message
    )  # raw provider text never reaches the client


async def test_error_after_partial_text_keeps_deltas_then_error():
    llm = FailingFake(exc=RuntimeError("dropped"), after_text="partial")

    events = [e async for e in stream_chat(make_request(), llm)]

    assert isinstance(events[0], Delta)
    assert isinstance(events[-1], ErrorEvent)
    assert not any(isinstance(e, Done) for e in events)


async def test_request_id_is_passed_through_and_generated():
    fixed = uuid4()
    events = [
        e
        async for e in stream_chat(make_request(), ProviderLikeFake(), request_id=fixed)
    ]
    assert events[-1].request_id == fixed

    generated = [e async for e in stream_chat(make_request(), ProviderLikeFake())]
    assert isinstance(generated[-1].request_id, UUID)
    assert generated[-1].request_id != fixed
