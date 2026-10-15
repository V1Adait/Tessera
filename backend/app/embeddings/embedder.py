"""Turn texts into vectors, behind one small interface:  await embedder.embed(texts) -> vectors.

`OpenRouterEmbedder` calls OpenRouter's OpenAI-compatible /embeddings endpoint, so one key covers
chat and embeddings and the model is a setting. Tests use a fake with the same interface.

Rules this file keeps:
  * One HTTP call per batch, retried with backoff on timeouts, 429 and 5xx (not on 4xx).
  * The reply is checked, never trusted: right count, same non-zero length, finite numbers.
  * Every call writes a trace row (kind 'embed') with tokens and cost when the provider gives them.
  * Errors carry a short code; the text of a provider error is logged, never shown to a user.
"""

import asyncio
import logging
import math
import time
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

import httpx

from backend.app.db.records import TraceRecord
from backend.app.tracing import Recorder, no_recorder

logger = logging.getLogger(__name__)

OPENROUTER_ROOT = "https://openrouter.ai/api/v1"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_ATTEMPTS = 3
_BACKOFF_S = 0.5


class EmbeddingError(Exception):
    """Embedding failed. `code` is one of: unavailable, auth, bad_request, bad_output, too_large."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Embedder(Protocol):
    name: str  # the model id; stored with the vectors so a model change is visible

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, same order. Raises EmbeddingError."""
        ...


def make_embedding_client(api_key: str, timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=OPENROUTER_ROOT,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )


def check_vectors(data: Any, expected: int) -> list[list[float]]:
    """Pull the vectors out of an OpenAI-style reply, in input order, and validate them."""
    try:
        items = sorted(data["data"], key=lambda item: item["index"])
        vectors = [[float(x) for x in item["embedding"]] for item in items]
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingError("bad_output", "unexpected embeddings response") from exc
    if len(vectors) != expected:
        raise EmbeddingError(
            "bad_output", f"asked for {expected} vectors, got {len(vectors)}"
        )
    width = len(vectors[0]) if vectors else 0
    if expected and width == 0:
        raise EmbeddingError("bad_output", "empty vector")
    for vector in vectors:
        if len(vector) != width:
            raise EmbeddingError("bad_output", "vectors of different lengths")
        if not all(math.isfinite(x) for x in vector):
            raise EmbeddingError("bad_output", "vector contains NaN or infinity")
    return vectors


class OpenRouterEmbedder:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        recorder: Recorder = no_recorder,
        backoff_s: float = _BACKOFF_S,
    ) -> None:
        self.name = model
        self._client = client
        self._recorder = recorder
        self._backoff_s = backoff_s

    async def _post(self, texts: list[str]) -> httpx.Response:
        last: EmbeddingError | None = None
        for attempt in range(_ATTEMPTS):
            if attempt:
                await asyncio.sleep(self._backoff_s * 2 ** (attempt - 1))
            try:
                response = await self._client.post(
                    "/embeddings",
                    json={
                        "model": self.name,
                        "input": texts,
                        "encoding_format": "float",
                    },
                )
            except httpx.TransportError as exc:  # timeouts, connection resets
                last = EmbeddingError("unavailable", type(exc).__name__)
                continue
            if response.status_code in _RETRY_STATUSES:
                last = EmbeddingError("unavailable", f"HTTP {response.status_code}")
                continue
            return response
        raise last or EmbeddingError("unavailable", "no response")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started = time.perf_counter()
        tokens: int | None = None
        cost: Decimal | None = None
        code: str | None = None
        try:
            response = await self._post(texts)
            if response.status_code != 200:
                code = "auth" if response.status_code in (401, 403) else "bad_request"
                logger.error(
                    "embeddings rejected: %s %s",
                    response.status_code,
                    response.text[:200],
                )
                raise EmbeddingError(code, f"HTTP {response.status_code}")
            data = response.json()
            usage = data.get("usage") or {}
            tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
            if usage.get("cost") is not None:
                cost = Decimal(str(usage["cost"]))
            return check_vectors(data, len(texts))
        except EmbeddingError as exc:
            code = exc.code
            raise
        except ValueError as exc:  # the body was not JSON
            code = "bad_output"
            raise EmbeddingError(code, "response was not JSON") from exc
        finally:
            await self._recorder(
                TraceRecord(
                    request_id=uuid4(),
                    kind="embed",
                    status="error" if code else "ok",
                    provider="openrouter",
                    model=self.name,
                    tokens_in=tokens,
                    cost_usd=cost,
                    cost_source="provider"
                    if cost is not None
                    else ("unknown" if tokens else None),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_code=code,
                )
            )


async def embed_in_batches(
    embedder: Embedder, texts: list[str], batch_size: int
) -> list[list[float]]:
    """Embed many texts in batches, in order. Stops at the first failed batch (the caller marks
    the document failed; nothing is half-saved because the vectors are stored only afterwards)."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), max(1, batch_size)):
        vectors += await embedder.embed(texts[start : start + batch_size])
    return vectors
