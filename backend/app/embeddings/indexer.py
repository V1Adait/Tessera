"""Keep the vector index in step with a document's chunks.

    claim the document (pending -> indexing, so two workers never index it at once)
      -> read its chunks from Postgres
      -> reuse vectors of text that did not change (same content_hash, same document);
         embed only the rest, in batches
      -> write the new points, THEN delete points of chunks that no longer exist
      -> record the outcome on the document: indexed, or failed with a plain reason

Failure is never silent and never fatal: the document stays searchable by full text, its
`index_status` says 'failed' and why, and a retry re-embeds only what is missing.
Nothing here runs inside the parse transaction, because embedding is a network call.
"""

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import asyncpg

from backend.app.config import Settings
from backend.app.db import repository as repo
from backend.app.db.records import ChunkHit, ChunkRecord
from backend.app.embeddings.embedder import Embedder, EmbeddingError, embed_in_batches
from backend.app.embeddings.search import vector_search
from backend.app.embeddings.store import QdrantStore, VectorPoint
from backend.app.retrieval.hybrid import HybridResult, hybrid_search, keyword_search
from backend.app.retrieval.sparse import SparseEncoder, SparseError

logger = logging.getLogger(__name__)

IndexOutcome = Literal["indexed", "failed", "skipped", "stale"]
# skipped: nothing to do (not pending, not parsed, or someone else has it)
# stale: finished, but the document changed meanwhile and will be indexed again

_REASONS = {
    "unavailable": "the embedding service could not be reached; retry later",
    "auth": "the embedding service rejected the API key",
    "bad_request": "the embedding service rejected the request",
    "bad_output": "the embedding service returned an unusable answer",
}


@dataclass(frozen=True)
class _Plan:
    to_embed: list[str]  # content hashes that need a vector
    texts: dict[str, str]  # hash -> text to embed


class Indexer:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        embedder: Embedder,
        store: QdrantStore,
        sparse: SparseEncoder,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._sparse = sparse
        self._slots = asyncio.Semaphore(
            max(1, settings.embedding_max_concurrent_indexes)
        )
        self._locks: dict[UUID, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def model(self) -> str:
        return self._embedder.name

    # --- scheduling -------------------------------------------------------------------------

    def schedule(self, document_id: UUID) -> None:
        """Index in the background. Safe to call often: a document that is not pending is skipped."""
        self._spawn(self.index_document(document_id))

    def _spawn(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def schedule_pending(self) -> int:
        """At startup: index every parsed document that is waiting. Returns how many."""
        ids = await repo.list_documents_to_index(self._pool)
        for document_id in ids:
            self.schedule(document_id)
        return len(ids)

    async def close(self, timeout: float = 10.0) -> None:
        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await self._store.close()

    async def search(
        self,
        workspace_id: UUID,
        query: str,
        *,
        entity_ids: list[UUID] | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        """Nearest chunks to a question inside a workspace. Raises EmbeddingError."""
        return await vector_search(
            self._pool,
            self._embedder,
            self._store,
            workspace_id,
            query,
            entity_ids=entity_ids,
            limit=limit,
        )

    async def hybrid_search(
        self,
        workspace_id: UUID,
        query: str,
        *,
        entity_ids: list[UUID] | None = None,
        limit: int = 10,
        **tuning: float,
    ) -> HybridResult:
        """BM25 keyword search + dense vector search merged with RRF. `tuning` passes k /
        dense_weight / sparse_weight through (the eval uses it). Degrades, never raises, when one
        side fails."""
        return await hybrid_search(
            self._pool,
            self._embedder,
            self._sparse,
            self._store,
            workspace_id,
            query,
            entity_ids=entity_ids,
            limit=limit,
            **tuning,  # type: ignore[arg-type]
        )

    async def keyword_search(
        self,
        workspace_id: UUID,
        query: str,
        *,
        entity_ids: list[UUID] | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        """BM25 alone: chunks that contain the question's words. Raises SparseError."""
        return await keyword_search(
            self._pool,
            self._sparse,
            self._store,
            workspace_id,
            query,
            entity_ids=entity_ids,
            limit=limit,
        )

    async def reindex(self, document_id: UUID) -> bool:
        """Index again from scratch of the document's state (for example after a model change).
        False if the document is unknown or being indexed right now."""
        if not await repo.mark_pending_index(self._pool, document_id):
            return False
        self.schedule(document_id)
        return True

    # --- one document -----------------------------------------------------------------------

    async def index_document(self, document_id: UUID) -> IndexOutcome:
        """Never raises (except for cancellation, which first puts the document back to pending)."""
        lock = self._locks.setdefault(document_id, asyncio.Lock())
        async with lock, self._slots:
            if not await repo.claim_for_indexing(self._pool, document_id):
                return "skipped"
            try:
                detail = await self._index(document_id)
            except EmbeddingError as exc:
                logger.warning(
                    "indexing %s failed: %s (%s)", document_id, exc.code, exc
                )
                reason = exc.message if exc.code == "too_large" else _REASONS[exc.code]
                return await self._finish(document_id, "failed", reason)
            except SparseError as exc:
                logger.warning(
                    "indexing %s failed: keyword encoder (%s)", document_id, exc.code
                )
                return await self._finish(
                    document_id,
                    "failed",
                    "the keyword encoder could not run; retry later",
                )
            except asyncio.CancelledError:
                await repo.release_indexing(self._pool, document_id)
                raise
            except Exception:
                logger.exception("unexpected error indexing document %s", document_id)
                return await self._finish(
                    document_id, "failed", "unexpected error while indexing"
                )
            return await self._finish(document_id, "indexed", detail)

    async def _finish(
        self,
        document_id: UUID,
        status: Literal["indexed", "failed"],
        detail: str | None,
    ) -> IndexOutcome:
        applied = await repo.set_index_result(
            self._pool, document_id, status, detail, self.model
        )
        return status if applied else "stale"

    async def _index(self, document_id: UUID) -> str | None:
        """Do the work. Returns a note for the document (or None). Raises EmbeddingError."""
        scope = await repo.document_scope(self._pool, document_id)
        if scope is None:
            return None
        workspace_id, entity_id = scope
        chunks = await repo.all_chunks(self._pool, document_id)
        limit = self._settings.embedding_max_chunks_per_document
        if len(chunks) > limit:
            raise _TooLarge(len(chunks), limit)

        stored_dim = await self._store.dimension()
        stored = (
            await self._store.document_points(stored_dim, document_id)
            if stored_dim
            else []
        )
        vectors, cut = await self._vectors_for(
            chunks, {p.content_hash: p.vector for p in stored}
        )
        # Keyword vectors are computed here, on this machine: no network, no cost, so there is
        # nothing worth reusing from the index.
        cap = self._settings.embedding_max_chars
        keywords = await self._sparse.encode_documents(
            [c.search_text[:cap] for c in chunks]
        )

        points = [
            VectorPoint(
                chunk_id=c.id,
                vector=vectors[c.content_hash],
                workspace_id=workspace_id,
                entity_id=entity_id,
                document_id=document_id,
                section_id=c.section_id,
                kind=c.kind,
                content_hash=c.content_hash,
                sparse=sparse,
            )
            for c, sparse in zip(chunks, keywords, strict=True)
        ]
        await self._store.upsert(points)
        keep = {c.id for c in chunks}
        stale = [p.chunk_id for p in stored if p.chunk_id not in keep]
        if stale and stored_dim:
            await self._store.delete(stored_dim, stale)
        return (
            f"{cut} chunk(s) were longer than the embedding limit and were cut for indexing"
            if cut
            else None
        )

    async def _vectors_for(
        self, chunks: list[ChunkRecord], reusable: dict[str, list[float]]
    ) -> tuple[dict[str, list[float]], int]:
        """hash -> vector for every chunk, embedding only what is not reusable. Also returns how
        many texts had to be cut to fit the embedding limit."""
        cap = self._settings.embedding_max_chars
        texts: dict[str, str] = {}
        cut = 0
        for chunk in chunks:
            if chunk.content_hash not in texts:
                texts[chunk.content_hash] = chunk.search_text[:cap]
                cut += len(chunk.search_text) > cap

        have = {h: v for h, v in reusable.items() if h in texts}
        for _ in range(
            2
        ):  # a second pass only if reused vectors have a different length
            missing = [h for h in texts if h not in have]
            if missing:
                fresh = await embed_in_batches(
                    self._embedder,
                    [texts[h] for h in missing],
                    self._settings.embedding_batch_size,
                )
                width = len(fresh[0])
                have = {h: v for h, v in have.items() if len(v) == width}
                have.update(zip(missing, fresh, strict=True))
            if all(h in have for h in texts):
                break
        return have, cut


class _TooLarge(EmbeddingError):
    """Refused before any embedding call: the document would cost more than the limit allows."""

    def __init__(self, count: int, limit: int) -> None:
        super().__init__(
            "too_large", f"too many chunks to index ({count}; the limit is {limit})"
        )
