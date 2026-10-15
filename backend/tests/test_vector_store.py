"""The Qdrant store: scope filters, reuse data, deletes. Every test runs twice: against the
embedded in-memory mode, and against a real Qdrant server when one is reachable at
TEST_QDRANT_URL (default http://127.0.0.1:6333). The server run skips itself if there is none.
Each server test uses its own collection prefix and removes it afterwards."""

import asyncio
import os
from uuid import uuid4

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from backend.app.embeddings.store import (
    LAYOUT,
    QdrantStore,
    VectorPoint,
    collection_name,
)
from backend.tests.fakes import FakeSparse, drop_test_collections

KEYWORDS = (
    FakeSparse()
)  # turns words into a sparse vector; Qdrant does the BM25 weighting
WS_A, WS_B = uuid4(), uuid4()
ACME, BETA = uuid4(), uuid4()
SERVER_URL = os.environ.get("TEST_QDRANT_URL", "http://127.0.0.1:6333")


def _server_up() -> bool:
    try:
        return httpx.get(f"{SERVER_URL}/healthz", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(params=["memory", "server"])
async def store(request):
    if request.param == "memory":
        client = AsyncQdrantClient(location=":memory:")
        s = QdrantStore(client, prefix="t", model="fake/model")
        yield s
        await s.close()
        return
    if not _server_up():
        pytest.skip(f"no Qdrant server at {SERVER_URL}")
    prefix = f"test_{uuid4().hex[:10]}"
    client = AsyncQdrantClient(url=SERVER_URL, timeout=60)
    s = QdrantStore(client, prefix=prefix, model="fake/model", payload_indexes=True)
    yield s
    await drop_test_collections(client, prefix)
    await s.close()


def point(vector, *, ws=WS_A, entity=ACME, document=None, content_hash="h", words=None):
    return VectorPoint(
        chunk_id=uuid4(),
        vector=vector,
        workspace_id=ws,
        entity_id=entity,
        document_id=document or uuid4(),
        section_id=uuid4(),
        kind="prose",
        content_hash=content_hash,
        sparse=KEYWORDS.vector(words) if words is not None else None,
    )


async def find(store, vector, *, ws=WS_A, entities=None, limit=5):
    return await store.search_dense(
        vector, workspace_id=ws, entity_ids=entities, limit=limit
    )


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


async def test_two_documents_written_at_the_same_moment_into_an_empty_store(store):
    """Both reach 'create the collection' together. This failed on a real server before the
    creation was serialised: the second writer got 'already exists'."""
    first, second = point([1.0, 0.0]), point([0.0, 1.0])

    await asyncio.gather(store.upsert([first]), store.upsert([second]))

    hits = await find(store, [1.0, 1.0], limit=5)
    assert {h.chunk_id for h in hits} == {first.chunk_id, second.chunk_id}


async def test_a_collection_someone_else_created_is_not_an_error(store):
    """A second process (or an earlier run) already made the collection."""
    await store.upsert([point([1.0, 0.0])])
    # same collection, but a store object that knows nothing yet (the prefix given to the
    # constructor is the original one: the layout tag is added inside)
    fresh = QdrantStore(
        store._client,
        prefix=store._prefix.removesuffix(f"_{LAYOUT}"),
        model=store._model,
    )

    await fresh.upsert([point([0.0, 1.0])])

    assert len(await find(fresh, [1.0, 1.0], limit=5)) == 2


# --- keyword (BM25) vectors ---------------------------------------------------------------


async def find_words(store, words, *, ws=WS_A, entities=None, limit=5):
    return await store.search_sparse(
        KEYWORDS.vector(words, query=True),
        workspace_id=ws,
        entity_ids=entities,
        limit=limit,
    )


def keyword_point(words, **kwargs):
    return point([1.0, 0.0], words=words, **kwargs)


async def test_a_chunk_is_found_by_a_word_it_contains(store):
    has, lacks = (
        keyword_point("revolver matures soon"),
        keyword_point("plants employ people"),
    )
    await store.upsert([has, lacks])

    hits = await find_words(store, "revolver")

    assert [h.chunk_id for h in hits] == [has.chunk_id]


async def test_a_rare_word_counts_for_more_than_a_common_one(store):
    common = [keyword_point(f"the company item{i}") for i in range(6)]
    rare = keyword_point("the company cupertino")
    await store.upsert([*common, rare])

    hits = await find_words(store, "company cupertino")

    assert hits[0].chunk_id == rare.chunk_id


async def test_keyword_search_is_scoped_to_the_workspace_and_entities(store):
    mine = keyword_point("revolver", ws=WS_A, entity=ACME)
    other_entity = keyword_point("revolver", ws=WS_A, entity=BETA)
    theirs = keyword_point("revolver", ws=WS_B, entity=ACME)
    await store.upsert([mine, other_entity, theirs])

    in_workspace = await find_words(store, "revolver")
    only_acme = await find_words(store, "revolver", entities=[ACME])

    assert {h.chunk_id for h in in_workspace} == {mine.chunk_id, other_entity.chunk_id}
    assert [h.chunk_id for h in only_acme] == [mine.chunk_id]


async def test_a_chunk_with_no_keywords_is_stored_and_found_by_meaning(store):
    no_words = keyword_point("the of and")  # only stopwords: an empty sparse vector
    never_set = point([1.0, 0.0])  # no sparse vector at all
    await store.upsert([no_words, never_set])

    by_meaning = await find(store, [1.0, 0.0])
    by_words = await find_words(store, "revolver")

    assert {h.chunk_id for h in by_meaning} == {no_words.chunk_id, never_set.chunk_id}
    assert by_words == []


async def test_an_empty_query_finds_nothing_and_does_not_raise(store):
    await store.upsert([keyword_point("revolver")])

    assert await find_words(store, "the of and") == []


async def test_keyword_search_before_anything_is_stored_returns_nothing(store):
    assert await find_words(store, "revolver") == []


async def test_keyword_scores_are_higher_for_better_matches(store):
    one = keyword_point("revolver matures")
    both = keyword_point("revolver revolver revolver facility")
    await store.upsert([one, both, keyword_point("unrelated text here")])

    hits = await find_words(store, "revolver facility")

    assert hits[0].chunk_id == both.chunk_id and hits[0].score > hits[1].score


async def test_the_dense_vector_is_still_read_back_for_reuse(store):
    document = uuid4()
    p = keyword_point("revolver", document=document)
    await store.upsert([p])

    [found] = await store.document_points(2, document)

    assert found.chunk_id == p.chunk_id and len(found.vector) == 2


async def test_collections_carry_the_layout_so_old_ones_are_never_reused(store):
    await store.upsert([keyword_point("revolver")])

    names = [c.name for c in (await store._client.get_collections()).collections]

    assert names and all("_hyb__" in n for n in names)
