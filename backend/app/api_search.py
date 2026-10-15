"""Search routes for looking at retrieval by hand: full text, or by meaning (vectors).

Thin handlers: the SQL is in the repository and the vector work in the Indexer. These return
chunks, not answers; nothing here calls a chat model.
"""

from typing import Annotated, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import ChunkHit
from backend.app.embeddings.embedder import EmbeddingError
from backend.app.embeddings.indexer import Indexer
from backend.app.retrieval.sparse import SparseError

router = APIRouter(tags=["search"])

PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    section_path: list[str]
    kind: str
    page_start: int | None
    score: float
    text: str  # the exact words from the document


def _hit(hit: ChunkHit) -> SearchHit:
    c = hit.chunk
    return SearchHit(
        chunk_id=c.id,
        document_id=c.document_id,
        filename=hit.filename,
        section_path=c.section_path,
        kind=c.kind,
        page_start=c.page_start,
        score=hit.score,
        text=c.text,
    )


def get_indexer(request: Request) -> Indexer | None:
    return getattr(request.app.state, "indexer", None)  # type: ignore[no-any-return]


IndexerDep = Annotated[Indexer | None, Depends(get_indexer)]


@router.get("/workspaces/{workspace_id}/search", response_model=list[SearchHit])
async def search(
    workspace_id: UUID,
    pool: PoolDep,
    indexer: IndexerDep,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    response: Response,
    mode: Literal["text", "vector", "bm25", "hybrid"] = "hybrid",
    entity_id: Annotated[list[UUID] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[SearchHit]:
    """Chunks matching `q` inside one workspace, optionally limited to some entities.

    mode=hybrid (the default) merges BM25 keyword results and vector results; if one side is
    down it returns the other's results and says so in the X-Retrieval-Degraded header.
    mode=text is the older Postgres full-text search, kept for comparison and as a last resort."""
    if await repo.get_workspace(pool, workspace_id) is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    if mode == "text":
        hits = await repo.search_chunks_text(
            pool, workspace_id, q, entity_ids=entity_id, limit=limit
        )
        return [_hit(h) for h in hits]
    if indexer is None:
        raise HTTPException(status_code=503, detail="vector search is not configured")
    if mode == "hybrid":
        result = await indexer.hybrid_search(
            workspace_id, q, entity_ids=entity_id, limit=limit
        )
        if result.degraded:
            response.headers["X-Retrieval-Degraded"] = result.degraded
        return [_hit(h) for h in result.hits]
    try:
        if mode == "bm25":
            found = await indexer.keyword_search(
                workspace_id, q, entity_ids=entity_id, limit=limit
            )
        else:
            found = await indexer.search(
                workspace_id, q, entity_ids=entity_id, limit=limit
            )
        return [_hit(h) for h in found]
    except EmbeddingError:
        raise HTTPException(
            status_code=503, detail="the embedding service is unavailable"
        ) from None
    except SparseError:
        raise HTTPException(
            status_code=503, detail="the keyword search is unavailable"
        ) from None
