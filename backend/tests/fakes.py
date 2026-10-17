"""Shared test doubles: a fake model that streams like a real provider, and test settings."""

import asyncio
import json
import math
import re
import zlib
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from backend.app.config import Settings
from backend.app.models import ChatRequest, Message
from backend.app.retrieval.sparse import SparseVec


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
    reported_cost: float | None = None

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
            response_metadata=(
                {"token_usage": {"cost": self.reported_cost}}
                if self.reported_cost is not None
                else {}
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
    reported_cost: float | None = None  # what the "provider" says each reply cost

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
            response_metadata=(
                {"token_usage": {"cost": self.reported_cost}}
                if self.reported_cost is not None
                else {}
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class HashEmbedder:
    """An embedder that needs no network: each word is hashed into a slot of a small vector,
    so texts that share words end up close (cosine). Records every batch it was asked for, and
    can be told to fail, or to wait on an event (to test what happens mid-indexing)."""

    def __init__(self, name: str = "fake/hash-embed", dim: int = 32) -> None:
        self.name = name
        self.dim = dim
        self.calls: list[list[str]] = []
        self.error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def vector(self, text: str) -> list[float]:
        slots = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            slots[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in slots)) or 1.0
        return [x / norm for x in slots]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return [self.vector(t) for t in texts]


async def drop_test_collections(client, prefix: str, attempts: int = 5) -> None:
    """Remove the throwaway collections a test made on a real Qdrant server.

    On Windows the server cannot rename a collection's folder while a background task still has a
    file in it open, and the delete fails with 'failed to rename file'. That is clean-up noise,
    not a test failure: retry briefly, and if it still will not go, leave it (a later run removes
    anything starting with the test prefix)."""
    for collection in (await client.get_collections()).collections:
        if not collection.name.startswith(prefix):
            continue
        for attempt in range(attempts):
            try:
                await client.delete_collection(collection.name)
                break
            except Exception:  # noqa: BLE001  # clean-up must never fail a test
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(0.5 * (attempt + 1))


class FakeSparse:
    """A keyword encoder that needs no model or network. Each word becomes a term (its hash) with
    its count as the weight; a query's terms all weigh 1. Qdrant applies the rarity weighting
    (IDF) itself, so ranking behaves like BM25. Can be told to fail."""

    name = "fake/bm25"
    STOPWORDS = frozenset(
        ["the", "a", "an", "of", "in", "is", "are", "to", "and", "what", "when", "does"]
    )

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.document_calls = 0

    def vector(self, text: str, query: bool = False) -> SparseVec:
        from collections import Counter

        words = [
            w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in self.STOPWORDS
        ]
        counts = Counter(zlib.crc32(w.encode()) for w in words)
        indices = sorted(counts)
        return SparseVec(indices, [1.0 if query else float(counts[i]) for i in indices])

    async def encode_documents(self, texts: list[str]) -> list[SparseVec]:
        self.document_calls += 1
        if self.error is not None:
            raise self.error
        return [self.vector(t) for t in texts]

    async def encode_query(self, text: str) -> SparseVec:
        if self.error is not None:
            raise self.error
        return self.vector(text, query=True)


class ToolFake(BaseChatModel):
    """A model that follows a script and can ask for tools, like a real tool-calling model.

    Each item of `steps` is one model call:
        "some text"                       -> a final answer
        [("tool_name", {"arg": 1}), ...]  -> asks for those tools (no text)
        ("some text", [("tool", {...})])  -> says something, then asks for tools
        an Exception                      -> the call fails
    When the script runs out it repeats the last item. Every prompt it was given is kept in `seen`.
    """

    steps: list[Any] = Field(default_factory=list)
    seen: list[Any] = Field(default_factory=list)
    tools_seen: list[Any] = Field(default_factory=list)
    reported_cost: float | None = None
    delay_s: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "tool-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolFake:  # type: ignore[override]
        self.tools_seen = list(tools)
        return self

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.seen.append(list(messages))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        item = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(item, Exception):
            raise item
        text, calls = "", []
        if isinstance(item, str):
            text = item
        elif isinstance(item, tuple):
            text, calls = item
        else:
            calls = item
        if text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        chunks = [
            {
                "name": name,
                "args": json.dumps(args),
                "id": f"call_{i}_{len(self.seen)}",
                "index": i,
            }
            for i, (name, args) in enumerate(calls)
        ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=chunks,  # type: ignore[arg-type]
                usage_metadata=UsageMetadata(
                    input_tokens=50, output_tokens=10, total_tokens=60
                ),
                response_metadata=(
                    {"token_usage": {"cost": self.reported_cost}}
                    if self.reported_cost is not None
                    else {}
                ),
            )
        )
