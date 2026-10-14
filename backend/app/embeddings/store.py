"""The vector store: Qdrant behind a small interface, so tests and a future swap stay cheap.

What lives where:
    Postgres  the chunks (text, search_text, offsets...). The source of truth.
    Qdrant    one vector per chunk, with just enough payload to FILTER by scope.
              The text is never copied here: a hit is a chunk id, and Postgres supplies the chunk.

Collections are named after the embedding model AND the vector length, so two models (or two
sizes of one model) can never be mixed in one index; changing the model starts a new collection.
A point's id is the chunk's id.
"""

import logging
import re
from dataclasses import dataclass
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


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


@dataclass(frozen=True)
class StoredPoint:
    chunk_id: UUID
    vector: list[float]
    content_hash: str


@dataclass(frozen=True)
class VectorHit:
    chunk_id: UUID
    score: float  # cosine similarity: higher is closer


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
        self._prefix = prefix
        self._model = model
        self._ready: set[str] = set()

    def _name(self, dimension: int) -> str:
        return collection_name(self._prefix, self._model, dimension)

    async def close(self) -> None:
        await self._client.close()

    async def dimension(self) -> int | None:
        """The vector length already in use for this model, if a collection exists."""
        stem = self._name(0).rsplit("__", 1)[0] + "__"
        names = sorted(
            c.name
            for c in (await self._client.get_collections()).collections
            if c.name.startswith(stem)
        )
        if not names:
            return None
        return int(names[0].rsplit("__", 1)[1])

    async def _ensure(self, dimension: int) -> str:
        name = self._name(dimension)
        if name in self._ready:
            return name
        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
            for field in (
                ("workspace_id", "entity_id", "document_id")
                if self._payload_indexes
                else ()
            ):
                try:  # server mode indexes the filter fields; embedded mode has nothing to index
                    await self._client.create_payload_index(
                        name, field, models.PayloadSchemaType.KEYWORD
                    )
                except Exception:  # noqa: BLE001  # filters still work without the index
                    logger.info("no payload index for %s on %s", field, name)
        self._ready.add(name)
        return name

    async def upsert(self, points: list[VectorPoint], *, batch_size: int = 256) -> None:
        if not points:
            return
        name = await self._ensure(len(points[0].vector))
        for start in range(0, len(points), batch_size):
            await self._client.upsert(
                name,
                points=[
                    models.PointStruct(
                        id=str(p.chunk_id),
                        vector=p.vector,
                        payload={
                            "workspace_id": str(p.workspace_id),
                            "entity_id": str(p.entity_id) if p.entity_id else None,
                            "document_id": str(p.document_id),
                            "section_id": str(p.section_id),
                            "kind": p.kind,
                            "content_hash": p.content_hash,
                        },
                    )
                    for p in points[start : start + batch_size]
                ],
            )

    async def document_points(
        self, dimension: int, document_id: UUID
    ) -> list[StoredPoint]:
        """Every stored point of one document, with its vector (so unchanged text is not
        embedded again)."""
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
                with_vectors=True,
            )
            for record in records:
                vector = record.vector
                payload = record.payload or {}
                if not isinstance(vector, list) or "content_hash" not in payload:
                    continue
                flat = [x for x in vector if isinstance(x, float | int)]
                if len(flat) != len(vector):  # a multi-vector point: not ours
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

    async def search(
        self,
        vector: list[float],
        *,
        workspace_id: UUID,
        entity_ids: list[UUID] | None,
        limit: int,
    ) -> list[VectorHit]:
        """Nearest chunks inside one workspace (and, if given, some entities). The scope is part
        of the search itself, not applied to its results."""
        name = self._name(len(vector))
        if not await self._client.collection_exists(name):
            return []
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
        result = await self._client.query_points(
            name, query=vector, query_filter=models.Filter(must=must), limit=limit
        )
        return [
            VectorHit(UUID(str(p.id)), float(p.score or 0.0)) for p in result.points
        ]


def make_qdrant_client(
    *, url: str | None, api_key: str | None, path: str
) -> AsyncQdrantClient:
    """A server if a URL is configured, otherwise embedded storage in a local folder."""
    if url:
        return AsyncQdrantClient(url=url, api_key=api_key)
    return AsyncQdrantClient(path=path)
