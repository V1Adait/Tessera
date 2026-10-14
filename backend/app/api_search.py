"""Search routes for looking at retrieval by hand: full text, or by meaning (vectors).

Thin handlers: the SQL is in the repository and the vector work in the Indexer. These return
chunks, not answers; nothing here calls a chat model.
"""

from typing import Annotated, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import ChunkHit
from backend.app.embeddings.embedder import EmbeddingError
from backend.app.embeddings.indexer import Indexer

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
    mode: Literal["text", "vector"] = "text",
    entity_id: Annotated[list[UUID] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[SearchHit]:
    """Chunks matching `q` inside one workspace, optionally limited to some entities."""
    if await repo.get_workspace(pool, workspace_id) is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    if mode == "text":
        hits = await repo.search_chunks_text(
            pool, workspace_id, q, entity_ids=entity_id, limit=limit
        )
        return [_hit(h) for h in hits]
    if indexer is None:
        raise HTTPException(status_code=503, detail="vector search is not configured")
    try:
        return [
            _hit(h)
            for h in await indexer.search(
                workspace_id, q, entity_ids=entity_id, limit=limit
            )
        ]
    except EmbeddingError:
        raise HTTPException(
            status_code=503, detail="the embedding service is unavailable"
        ) from None
