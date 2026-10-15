"""Sparse (keyword) vectors: text -> a short list of (term id, weight) pairs, for BM25 search.

A sparse vector says "these terms appear, this strongly". Qdrant stores them next to the dense
vector and, with its IDF setting on, weights each term by how rare it is across the collection,
which is what makes BM25 rank "Cupertino" above "company". So this file only does the part
Qdrant cannot: turn text into terms (lower-case, split into words, drop stopwords, stem, hash)
and weight each term by its frequency in THIS chunk.

    Bm25Encoder   the real one: fastembed's `Qdrant/bm25` model. It runs on this machine, costs
                  nothing per use, and needs a one-off download (a few small files, cached).
    SparseEncoder the interface; tests use a fake with the same shape.

Term ids are hashes, so no vocabulary is stored and two documents agree on ids without sharing
anything. Chunk and query use different weights on purpose: a chunk's weights carry term
frequency and length, a query's are all 1 (each term just says "I want this").
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]

    @property
    def empty(self) -> bool:
        return not self.indices


class SparseError(Exception):
    """The keyword encoder could not run (model missing and no network, for example)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SparseEncoder(Protocol):
    name: str

    async def encode_documents(self, texts: list[str]) -> list[SparseVec]:
        """One vector per text, same order. A text with no usable terms gets an empty vector."""
        ...

    async def encode_query(self, text: str) -> SparseVec:
        """The question as a vector. Empty if it holds only stopwords."""
        ...


class Bm25Encoder:
    """fastembed's BM25. The model object is created on first use, off the event loop, and all
    calls go through one lock (the underlying tokenizer is not documented as thread-safe)."""

    def __init__(
        self, model: str = "Qdrant/bm25", cache_dir: str | None = None
    ) -> None:
        self.name = model
        self._cache_dir = cache_dir
        self._model: object | None = None
        self._lock = asyncio.Lock()

    def _load(self) -> object:
        from fastembed import SparseTextEmbedding

        if self._cache_dir:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        return SparseTextEmbedding(model_name=self.name, cache_dir=self._cache_dir)

    async def _run(self, method: str, texts: list[str]) -> list[SparseVec]:
        async with self._lock:
            try:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load)
                model = self._model
                found = await asyncio.to_thread(
                    lambda: list(getattr(model, method)(texts))
                )
            except Exception as exc:  # a download, an import or a model error
                logger.exception("BM25 encoder failed: %s", type(exc).__name__)
                raise SparseError(
                    "unavailable", "the keyword encoder could not run"
                ) from exc
        return [
            SparseVec([int(i) for i in e.indices], [float(v) for v in e.values])
            for e in found
        ]

    async def encode_documents(self, texts: list[str]) -> list[SparseVec]:
        return await self._run("embed", texts) if texts else []

    async def encode_query(self, text: str) -> SparseVec:
        [vector] = await self._run("query_embed", [text])
        return vector
