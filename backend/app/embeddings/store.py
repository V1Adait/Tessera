"""The vector store: Qdrant behind a small interface, so tests and a future swap stay cheap.

What lives where:
    Postgres  the chunks (text, search_text, offsets...). The source of truth.
    Qdrant    two vectors per chunk, with just enough payload to FILTER by scope:
                "dense"  what the chunk MEANS (an embedding, searched by cosine similarity)
                "bm25"   which words it HAS (a sparse vector; Qdrant weights each word by how rare
                         it is across the collection, which is BM25)
              The text is never copied here: a hit is a chunk id, and Postgres supplies the chunk.

Collections are named after the layout, the embedding model AND the vector length, so two models
(or two sizes of one model, or the older single-vector layout) can never be mixed in one index;
changing any of them starts a new collection. A point's id is the chunk's id.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.app.retrieval.sparse import SparseVec

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")
DENSE = "dense"  # names of the two vectors in every point
SPARSE = "bm25"
LAYOUT = "hyb"  # collections with a dense AND a sparse vector (the first layout had only one)


@dataclass(frozen=True)
class VectorPoint:
    chunk_id: UUID
    vector: list[float]
    workspace_id: UUID
    entity_id: UUID | None
    document_id: UUID
    section_id: UUID
    kind: str
    content_hash: str
    sparse: SparseVec | None = (
        None  # None or empty: the chunk has no searchable keywords
    )


@dataclass(frozen=True)
class StoredPoint:
    chunk_id: UUID
    vector: list[float]
    content_hash: str


@dataclass(frozen=True)
class VectorHit:
    chunk_id: UUID
    score: float  # dense: cosine similarity; sparse: BM25-style score. Higher is better


def collection_name(prefix: str, model: str, dimension: int) -> str:
    slug = _SLUG.sub("_", model.lower()).strip("_")
    return f"{prefix}__{slug}__{dimension}"


class QdrantStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        prefix: str,
        model: str,
        payload_indexes: bool = False,
    ) -> None:
        """`payload_indexes` speeds up the scope filters on a real server; embedded mode has none."""
        self._client = client
        self._payload_indexes = payload_indexes
        self._prefix = f"{prefix}_{LAYOUT}"
        self._model = model
        self._dimension: int | None = None
        self._ready: set[str] = set()
        self._create_lock = asyncio.Lock()

    def _name(self, dimension: int) -> str:
        return collection_name(self._prefix, self._model, dimension)

    async def close(self) -> None:
        await self._client.close()

    async def dimension(self) -> int | None:
        """The vector length already in use for this model, if a collection exists."""
        if self._dimension is not None:
            return self._dimension
        stem = self._name(0).rsplit("__", 1)[0] + "__"
        names = sorted(
            c.name
            for c in (await self._client.get_collections()).collections
            if c.name.startswith(stem)
        )
        if not names:
            return None
        self._dimension = int(names[0].rsplit("__", 1)[1])
        return self._dimension

    async def _ensure(self, dimension: int) -> str:
        """Make sure the collection exists. Two documents indexed at the same moment both reach
        this point: the lock makes the second wait and find it already created, and a 409
        ("already exists") from another process is treated as success."""
        name = self._name(dimension)
        if name in self._ready:
            return name
        async with self._create_lock:
            if name in self._ready:
                return name
            if not await self._client.collection_exists(name):
                try:
                    await self._client.create_collection(
                        name,
                        vectors_config={
                            DENSE: models.VectorParams(
                                size=dimension, distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            # IDF is the "inverse document frequency" half of BM25: a word found
                            # in few chunks counts for more than one found everywhere.
                            SPARSE: models.SparseVectorParams(
                                modifier=models.Modifier.IDF
                            )
                        },
                    )
                except UnexpectedResponse as exc:
                    if exc.status_code != 409:
                        raise
                    logger.info("collection %s was created by someone else", name)
                for field in (
                    ("workspace_id", "entity_id", "document_id")
                    if self._payload_indexes
                    else ()
                ):
                    try:  # a real server indexes the filter fields; embedded mode has none
                        await self._client.create_payload_index(
                            name, field, models.PayloadSchemaType.KEYWORD
                        )
                    except Exception:  # noqa: BLE001  # filters still work without the index
                        logger.info("no payload index for %s on %s", field, name)
            self._ready.add(name)
            self._dimension = dimension
        return name

    @staticmethod
    def _struct(p: VectorPoint) -> models.PointStruct:
        vector: dict[str, Any] = {DENSE: p.vector}
        if (
            p.sparse is not None and not p.sparse.empty
        ):  # an empty sparse vector is left out
            vector[SPARSE] = models.SparseVector(
                indices=p.sparse.indices, values=p.sparse.values
            )
        return models.PointStruct(
            id=str(p.chunk_id),
            vector=vector,
            payload={
                "workspace_id": str(p.workspace_id),
                "entity_id": str(p.entity_id) if p.entity_id else None,
                "document_id": str(p.document_id),
                "section_id": str(p.section_id),
                "kind": p.kind,
                "content_hash": p.content_hash,
            },
        )

    async def upsert(self, points: list[VectorPoint], *, batch_size: int = 256) -> None:
        if not points:
            return
        name = await self._ensure(len(points[0].vector))
        for start in range(0, len(points), batch_size):
            await self._client.upsert(
                name,
                points=[self._struct(p) for p in points[start : start + batch_size]],
            )

    async def document_points(
        self, dimension: int, document_id: UUID
    ) -> list[StoredPoint]:
        """Every stored point of one document, with its dense vector (so unchanged text is not
        embedded again). Sparse vectors are not read back: they cost nothing to rebuild."""
        name = self._name(dimension)
        if not await self._client.collection_exists(name):
            return []
        found: list[StoredPoint] = []
        offset = None
        while True:
            records, offset = await self._client.scroll(
                name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=[DENSE],
            )
            for record in records:
                vectors = record.vector
                payload = record.payload or {}
                dense = vectors.get(DENSE) if isinstance(vectors, dict) else None
                if not isinstance(dense, list) or "content_hash" not in payload:
                    continue
                flat = [x for x in dense if isinstance(x, float | int)]
                if len(flat) != len(dense):  # a multi-vector point: not ours
                    continue
                found.append(
                    StoredPoint(
                        UUID(str(record.id)),
                        [float(x) for x in flat],
                        payload["content_hash"],
                    )
                )
            if offset is None:
                return found

    async def delete(self, dimension: int, chunk_ids: list[UUID]) -> None:
        name = self._name(dimension)
        if not chunk_ids or not await self._client.collection_exists(name):
            return
        await self._client.delete(
            name,
            points_selector=models.PointIdsList(points=[str(i) for i in chunk_ids]),
        )

    @staticmethod
    def _scope(workspace_id: UUID, entity_ids: list[UUID] | None) -> models.Filter:
        """The scope is part of the search itself, not applied to its results."""
        must: list[models.Condition] = [
            models.FieldCondition(
                key="workspace_id", match=models.MatchValue(value=str(workspace_id))
            )
        ]
        if entity_ids is not None:
            must.append(
                models.FieldCondition(
                    key="entity_id",
                    match=models.MatchAny(any=[str(e) for e in entity_ids]),
                )
            )
        return models.Filter(must=must)

    async def search_dense(
        self,
        vector: list[float],
        *,
        workspace_id: UUID,
        entity_ids: list[UUID] | None,
        limit: int,
    ) -> list[VectorHit]:
        """Chunks whose MEANING is nearest, inside one workspace (and, if given, some entities)."""
        name = self._name(len(vector))
        if not await self._client.collection_exists(name):
            return []
        result = await self._client.query_points(
            name,
            query=vector,
            using=DENSE,
            query_filter=self._scope(workspace_id, entity_ids),
            limit=limit,
        )
        return [
            VectorHit(UUID(str(p.id)), float(p.score or 0.0)) for p in result.points
        ]

    async def search_sparse(
        self,
        sparse: SparseVec,
        *,
        workspace_id: UUID,
        entity_ids: list[UUID] | None,
        limit: int,
    ) -> list[VectorHit]:
        """Chunks that contain the query's WORDS, rarest words counting most (BM25)."""
        dimension = await self.dimension()
        if sparse.empty or dimension is None:
            return []
        result = await self._client.query_points(
            self._name(dimension),
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=SPARSE,
            query_filter=self._scope(workspace_id, entity_ids),
            limit=limit,
        )
        return [
            VectorHit(UUID(str(p.id)), float(p.score or 0.0)) for p in result.points
        ]


def make_qdrant_client(
    *, url: str | None, api_key: str | None, path: str, timeout_s: float = 30.0
) -> AsyncQdrantClient:
    """A server if a URL is configured, otherwise embedded storage in a local folder. The client's
    own default timeout is 5 seconds, too short for a big batch on a busy server."""
    if url:
        return AsyncQdrantClient(url=url, api_key=api_key, timeout=int(timeout_s))
    return AsyncQdrantClient(path=path)
