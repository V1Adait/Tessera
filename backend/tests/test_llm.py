import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from backend.app.llm import stream_chat, to_lc_messages
from backend.app.models import ChatRequest, Delta, Done


def make_request(**kwargs) -> ChatRequest:
    return ChatRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)


class ProviderLikeFake(BaseChatModel):
    """Streams like a real provider: empty leading chunks, text, then usage on an empty chunk."""

    @property
    def _llm_type(self) -> str:
        return "provider-like-fake"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        for text in ["", "", "hello", " there"]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        usage = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))  # trailing chunk, no usage


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

    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in msgs] == ["be brief", "hi", "yo", "again"]


def test_to_lc_messages_without_system_prompt():
    msgs = to_lc_messages(make_request())

    assert [type(m) for m in msgs] == [HumanMessage]


def test_to_lc_messages_maps_system_role_in_list():
    req = ChatRequest(messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}])

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
