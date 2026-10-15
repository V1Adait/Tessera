"""Vector search: question -> embedding -> nearest chunks inside the scope -> ChunkHit.

The result type is the same ChunkHit the full-text search returns, so parent expansion
(chunking/expand.py) works on either. A vector hit is only a chunk id: the chunk itself is
loaded from Postgres, filtered to the workspace once more, and a hit whose chunk no longer exists
(an index slightly behind its chunks) is simply dropped.
"""

import logging
from uuid import UUID

import asyncpg

from backend.app.db import repository as repo
from backend.app.db.records import ChunkHit
from backend.app.embeddings.embedder import Embedder
from backend.app.embeddings.store import QdrantStore

logger = logging.getLogger(__name__)


async def vector_search(
    pool: asyncpg.Pool,
    embedder: Embedder,
    store: QdrantStore,
    workspace_id: UUID,
    query: str,
    *,
    entity_ids: list[UUID] | None = None,
    limit: int = 10,
) -> list[ChunkHit]:
    """Raises EmbeddingError if the query cannot be embedded."""
    [vector] = await embedder.embed([query])
    found = await store.search_dense(
        vector, workspace_id=workspace_id, entity_ids=entity_ids, limit=limit
    )
    if not found:
        return []
    chunks = await repo.chunks_in_workspace(
        pool, workspace_id, [h.chunk_id for h in found]
    )
    if len(chunks) < len(found):
        logger.info(
            "%s vector hit(s) had no chunk in this workspace; dropped",
            len(found) - len(chunks),
        )
    return [
        ChunkHit(
            chunk=chunks[h.chunk_id][0], filename=chunks[h.chunk_id][1], score=h.score
        )
        for h in found
        if h.chunk_id in chunks
    ]
