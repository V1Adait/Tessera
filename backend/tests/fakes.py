"""Shared test doubles: a fake model that streams like a real provider, and test settings."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from backend.app.config import Settings
from backend.app.models import ChatRequest, Message


class ScriptedFake(BaseChatModel):
    """Streams 'hello there' then usage, like a provider. Records the prompts it was given."""

    seen: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.seen.append(messages)
        for text in ["hello", " there"]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        usage = UsageMetadata(input_tokens=5, output_tokens=2, total_tokens=7)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=usage)
        )


def make_settings(max_context_tokens: int = 100_000) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]  # never read the real .env in tests
        provider="anthropic",
        anthropic_api_key="k",
        default_model="m",
        max_tokens_ceiling=4096,
        default_max_tokens=1000,
        request_timeout=10,
        max_context_tokens=max_context_tokens,
    )


def request(text: str = "hi") -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content=text)])


class SlowFake(BaseChatModel):
    """Streams five words with a pause between them, so a client can disconnect mid-stream."""

    @property
    def _llm_type(self) -> str:
        return "slow-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        for word in ["one ", "two ", "three ", "four ", "five "]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
            await asyncio.sleep(0.05)
        usage = UsageMetadata(input_tokens=5, output_tokens=5, total_tokens=10)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=usage)
        )


class BoomFake(BaseChatModel):
    """Fails after sending one chunk, like a provider dropping mid-answer."""

    @property
    def _llm_type(self) -> str:
        return "boom-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="partial "))
        raise RuntimeError("provider exploded with secret detail")


class SummaryFake(BaseChatModel):
    """Stands in for the cheap summariser. Answers with `reply`, records every prompt it was
    given, and can raise `error` instead (to test failures)."""

    reply: str = "- user is researching Visa\n- prefers short answers"
    error: Exception | None = None
    seen: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "summary-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(messages)
        if self.error is not None:
            raise self.error
        message = AIMessage(
            content=self.reply,
            usage_metadata=UsageMetadata(
                input_tokens=300, output_tokens=40, total_tokens=340
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class BigPromptFake(BaseChatModel):
    """Answers 'ok' and reports that the prompt it received was big (2800 tokens), the way a
    provider does, so the summary threshold is crossed."""

    @property
    def _llm_type(self) -> str:
        return "big-prompt-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))
        usage = UsageMetadata(input_tokens=2800, output_tokens=1, total_tokens=2801)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=usage)
        )


class QueueFake(BaseChatModel):
    """Answers with the next item of `replies` (a str, or an Exception to raise), repeating the
    last one when the list runs out. Records every prompt, so tests can check what was sent."""

    replies: list[Any] = Field(default_factory=list)
    seen: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "queue-fake"

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(messages)
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, Exception):
            raise item
        message = AIMessage(
            content=item,
            usage_metadata=UsageMetadata(
                input_tokens=50, output_tokens=10, total_tokens=60
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
