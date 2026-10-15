"""Hybrid search: keyword (BM25) and meaning (dense vectors) run together, merged with RRF.

    question ─┬─ BM25: which chunks contain these words? (rare words count most) ──┐
              │                                                                    ├─ RRF ─> chunks
              └─ dense: which chunks MEAN something close to this? ───────────────┘

Both legs query the same Qdrant collection, each scoped to the workspace (and entities) inside
its own query, and each fetches `candidates` chunks, so a chunk that is 30th in one list can still
be lifted by being 3rd in the other. The chunks themselves are then loaded from Postgres, which
checks the workspace once more.

Failure is never silent, and one failing leg never fails the search:
    dense leg fails (embedding service down)   -> keyword results only,  `degraded` says so
    keyword leg fails (encoder cannot run)     -> meaning results only,   `degraded` says so
    both fail (the vector index is unreachable) -> Postgres full text,    `degraded` says so
Only if that last resort fails too does the error reach the caller.
"""

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

import asyncpg

from backend.app.db import repository as repo
from backend.app.db.records import ChunkHit
from backend.app.embeddings.embedder import Embedder, EmbeddingError
from backend.app.embeddings.store import QdrantStore, VectorHit
from backend.app.retrieval.fusion import rrf
from backend.app.retrieval.sparse import SparseEncoder, SparseError
from backend.app.retrieval.terms import or_query

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = 50


@dataclass(frozen=True)
class HybridResult:
    hits: list[ChunkHit]
    degraded: str | None  # None, or why only part of the search ran
    sparse_found: int  # how many candidates each leg contributed
    dense_found: int


async def _hydrate(
    pool: asyncpg.Pool,
    workspace_id: UUID,
    ranked: list[tuple[UUID, float]],
) -> list[ChunkHit]:
    """Chunk rows for (chunk id, score) pairs, in the same order. Hits whose chunk is gone, or
    belongs to another workspace, are dropped."""
    if not ranked:
        return []
    chunks = await repo.chunks_in_workspace(pool, workspace_id, [i for i, _ in ranked])
    if len(chunks) < len(ranked):
        logger.info(
            "%s hit(s) had no chunk in this workspace; dropped",
            len(ranked) - len(chunks),
        )
    return [
        ChunkHit(chunk=chunks[i][0], filename=chunks[i][1], score=score)
        for i, score in ranked
        if i in chunks
    ]


async def _sparse_leg(
    encoder: SparseEncoder,
    store: QdrantStore,
    workspace_id: UUID,
    query: str,
    entity_ids: list[UUID] | None,
    candidates: int,
) -> list[VectorHit]:
    vector = await encoder.encode_query(query)
    return await store.search_sparse(
        vector, workspace_id=workspace_id, entity_ids=entity_ids, limit=candidates
    )


async def _dense_leg(
    embedder: Embedder,
    store: QdrantStore,
    workspace_id: UUID,
    query: str,
    entity_ids: list[UUID] | None,
    candidates: int,
) -> list[VectorHit]:
    [vector] = await embedder.embed([query])
    return await store.search_dense(
        vector, workspace_id=workspace_id, entity_ids=entity_ids, limit=candidates
    )


async def keyword_search(
    pool: asyncpg.Pool,
    encoder: SparseEncoder,
    store: QdrantStore,
    workspace_id: UUID,
    query: str,
    *,
    entity_ids: list[UUID] | None = None,
    limit: int = 10,
) -> list[ChunkHit]:
    """BM25 alone. Raises SparseError if the encoder cannot run."""
    found = await _sparse_leg(encoder, store, workspace_id, query, entity_ids, limit)
    return await _hydrate(pool, workspace_id, [(h.chunk_id, h.score) for h in found])


def _why(leg: str, error: BaseException) -> str:
    if isinstance(error, EmbeddingError | SparseError):
        return f"{leg} unavailable ({error.code})"
    return f"{leg} unavailable (index error)"


async def hybrid_search(
    pool: asyncpg.Pool,
    embedder: Embedder,
    encoder: SparseEncoder,
    store: QdrantStore,
    workspace_id: UUID,
    query: str,
    *,
    entity_ids: list[UUID] | None = None,
    limit: int = 10,
    candidates: int = DEFAULT_CANDIDATES,
    k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> HybridResult:
    sparse_result, dense_result = await asyncio.gather(
        _sparse_leg(encoder, store, workspace_id, query, entity_ids, candidates),
        _dense_leg(embedder, store, workspace_id, query, entity_ids, candidates),
        return_exceptions=True,
    )
    for result in (sparse_result, dense_result):
        if isinstance(result, asyncio.CancelledError):
            raise result

    sparse_hits = [] if isinstance(sparse_result, BaseException) else sparse_result
    dense_hits = [] if isinstance(dense_result, BaseException) else dense_result

    if isinstance(sparse_result, BaseException) and isinstance(
        dense_result, BaseException
    ):
        return await _last_resort(
            pool, workspace_id, query, entity_ids, limit, dense_result
        )

    degraded: str | None = None
    if isinstance(dense_result, BaseException):
        degraded = f"{_why('vector search', dense_result)}: keyword results only"
        logger.warning("hybrid search degraded: %s", degraded)
    elif isinstance(sparse_result, BaseException):
        degraded = f"{_why('keyword search', sparse_result)}: vector results only"
        logger.warning("hybrid search degraded: %s", degraded)

    fused = rrf(
        [[h.chunk_id for h in sparse_hits], [h.chunk_id for h in dense_hits]],
        k=k,
        weights=[sparse_weight, dense_weight],
    )
    hits = await _hydrate(pool, workspace_id, fused[:limit])
    return HybridResult(hits, degraded, len(sparse_hits), len(dense_hits))


async def _last_resort(
    pool: asyncpg.Pool,
    workspace_id: UUID,
    query: str,
    entity_ids: list[UUID] | None,
    limit: int,
    error: BaseException,
) -> HybridResult:
    """Both Qdrant legs failed, so the index itself is out of reach. Postgres can still answer a
    keyword question from the chunks it holds, less well. The caller is told."""
    logger.error("vector index unreachable; using Postgres full text", exc_info=error)
    terms = or_query(query)
    hits = (
        await repo.search_chunks_text(
            pool, workspace_id, terms, entity_ids=entity_ids, limit=limit
        )
        if terms
        else []
    )
    return HybridResult(
        hits,
        "vector index unavailable: database full-text results only",
        len(hits),
        0,
    )
