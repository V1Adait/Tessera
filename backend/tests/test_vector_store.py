"""The Qdrant store in embedded (in-memory) mode: scope filters, reuse data, deletes."""

from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from backend.app.embeddings.store import QdrantStore, VectorPoint, collection_name

WS_A, WS_B = uuid4(), uuid4()
ACME, BETA = uuid4(), uuid4()


@pytest.fixture
async def store():
    client = AsyncQdrantClient(location=":memory:")
    s = QdrantStore(client, prefix="t", model="fake/model")
    yield s
    await s.close()


def point(vector, *, ws=WS_A, entity=ACME, document=None, content_hash="h"):
    return VectorPoint(
        chunk_id=uuid4(),
        vector=vector,
        workspace_id=ws,
        entity_id=entity,
        document_id=document or uuid4(),
        section_id=uuid4(),
        kind="prose",
        content_hash=content_hash,
    )


async def find(store, vector, *, ws=WS_A, entities=None, limit=5):
    return await store.search(vector, workspace_id=ws, entity_ids=entities, limit=limit)


def test_collections_are_named_after_the_model_and_the_vector_length():
    a = collection_name("chunks", "openai/text-embedding-3-small", 1536)
    b = collection_name("chunks", "openai/text-embedding-3-small", 512)
    c = collection_name("chunks", "baai/bge-m3", 1536)

    assert len({a, b, c}) == 3
    assert a == "chunks__openai_text_embedding_3_small__1536"


async def test_the_nearest_point_comes_first(store):
    near, far = point([1.0, 0.0]), point([0.0, 1.0])
    await store.upsert([far, near])

    hits = await find(store, [0.9, 0.1])

    assert [h.chunk_id for h in hits] == [near.chunk_id, far.chunk_id]
    assert hits[0].score > hits[1].score


async def test_another_workspaces_points_never_appear(store):
    mine, theirs = point([1.0, 0.0], ws=WS_A), point([1.0, 0.0], ws=WS_B)
    await store.upsert([mine, theirs])

    assert [h.chunk_id for h in await find(store, [1.0, 0.0])] == [mine.chunk_id]


async def test_the_entity_filter_limits_the_search_and_none_means_all(store):
    a = point([1.0, 0.0], entity=ACME)
    b = point([1.0, 0.0], entity=BETA)
    loose = point([1.0, 0.0], entity=None)
    await store.upsert([a, b, loose])

    only_acme = await find(store, [1.0, 0.0], entities=[ACME])
    both = await find(store, [1.0, 0.0], entities=[ACME, BETA])
    everything = await find(store, [1.0, 0.0])

    assert {h.chunk_id for h in only_acme} == {a.chunk_id}
    assert {h.chunk_id for h in both} == {a.chunk_id, b.chunk_id}
    assert len(everything) == 3


async def test_searching_before_anything_is_stored_returns_nothing(store):
    assert await find(store, [1.0, 0.0]) == []
    assert await store.dimension() is None


async def test_the_stored_dimension_is_discovered(store):
    await store.upsert([point([1.0, 0.0, 0.0])])

    assert await store.dimension() == 3


async def test_a_documents_points_come_back_with_vectors_and_hashes(store):
    document = uuid4()
    p = point([0.5, 0.5], document=document, content_hash="abc")
    await store.upsert([p, point([1.0, 0.0])])  # another document's point

    found = await store.document_points(2, document)

    assert [(f.chunk_id, f.content_hash) for f in found] == [(p.chunk_id, "abc")]
    # Qdrant keeps cosine vectors at unit length; the direction (all that cosine uses) is intact
    assert found[0].vector == pytest.approx([0.7071, 0.7071], abs=1e-3)


async def test_deleting_removes_only_those_points(store):
    keep, drop = point([1.0, 0.0]), point([1.0, 0.0])
    await store.upsert([keep, drop])

    await store.delete(2, [drop.chunk_id])

    assert [h.chunk_id for h in await find(store, [1.0, 0.0])] == [keep.chunk_id]


async def test_upserting_the_same_chunk_twice_keeps_one_point(store):
    p = point([1.0, 0.0])
    await store.upsert([p])
    await store.upsert([p])

    assert len(await find(store, [1.0, 0.0])) == 1


async def test_deleting_from_a_missing_collection_is_not_an_error(store):
    await store.delete(2, [uuid4()])

    assert await store.document_points(2, uuid4()) == []
