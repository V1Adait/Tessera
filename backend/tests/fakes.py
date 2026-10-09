"""Shared test doubles: a fake model that streams like a real provider, and test settings."""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
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
