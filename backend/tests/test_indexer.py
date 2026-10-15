"""Indexing chunks into the vector store, against a real database and Qdrant (in memory, and a real server when one is up).

Each test gets a throwaway schema (conftest) and skips without a database. The embedder is a fake
(HashEmbedder: shared words give nearby vectors), so nothing touches the network.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from backend.app.api_documents import get_documents
from backend.app.api_search import get_indexer
from backend.app.config import get_settings
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.documents_service import DocumentService
from backend.app.embeddings.embedder import EmbeddingError
from backend.app.embeddings.indexer import Indexer
from backend.app.embeddings.store import QdrantStore
from backend.app.main import app
from backend.tests import make_files as mf
from backend.tests.fakes import HashEmbedder, make_settings
from backend.tests.test_chunks_db import reader, wait_final

FINAL = {"parsed", "needs_review", "needs_ocr", "failed"}
SERVER_URL = os.environ.get("TEST_QDRANT_URL", "http://127.0.0.1:6333")


def server_up() -> bool:
    try:
        return httpx.get(f"{SERVER_URL}/healthz", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture
def settings(tmp_path):
    return make_settings().model_copy(update={"upload_dir": str(tmp_path / "uploads")})


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture(params=["memory", "server"])
async def store(request):
    """Every indexer test runs on embedded Qdrant and, when one is reachable, on a real server
    (TEST_QDRANT_URL, default http://127.0.0.1:6333) in its own throwaway collections."""
    if request.param == "memory":
        s = QdrantStore(
            AsyncQdrantClient(location=":memory:"), prefix="t", model="fake/hash-embed"
        )
        yield s
        await s.close()
        return
    if not server_up():
        pytest.skip(f"no Qdrant server at {SERVER_URL}")
    prefix = f"test_{uuid4().hex[:10]}"
    client = AsyncQdrantClient(url=SERVER_URL, timeout=60)
    s = QdrantStore(
        client, prefix=prefix, model="fake/hash-embed", payload_indexes=True
    )
    yield s
    for c in (await client.get_collections()).collections:
        if c.name.startswith(prefix):
            await client.delete_collection(c.name)
    await s.close()


@pytest.fixture
async def indexer(pool, settings, embedder, store):
    ix = Indexer(pool, settings, embedder, store)
    yield ix
    for task in list(ix._tasks):
        task.cancel()
    await asyncio.gather(*ix._tasks, return_exceptions=True)


@pytest.fixture
async def service(pool, settings):
    """A document service WITHOUT an indexer: the test decides when indexing happens."""
    svc = DocumentService(pool, settings)
    yield svc
    await svc.close()


def a_docx(*paragraphs: str, heading: str = "Liquidity") -> bytes:
    def build(d):
        d.add_heading(heading, 1)
        for text in paragraphs:
            d.add_paragraph(text)

    return mf.docx_bytes(build)


async def parsed(service, pool, data: bytes, name="report.docx", entity_id=None):
    result = await service.upload(
        workspace_id=None, entity_id=entity_id, filename=name, read=reader(data)
    )
    return await wait_final(pool, result.document.id)


async def wait_index(pool, document_id, wanted, timeout=15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        document = await repo.get_document(pool, document_id)
        assert document is not None
        if document.index_status in wanted:
            return document
        await asyncio.sleep(0.05)
    raise AssertionError(f"index_status stayed {document.index_status!r}")


async def stored(store, pool, document_id):
    dim = await store.dimension()
    return await store.document_points(dim, document_id) if dim else []


# --- the happy path -----------------------------------------------------------------------


async def test_a_parsed_document_starts_pending_and_indexing_makes_it_indexed(
    pool, service, indexer, store
):
    document = await parsed(
        service, pool, a_docx("The revolver matures in March 2027.")
    )
    assert document.index_status == "pending" and document.indexed_model is None

    outcome = await indexer.index_document(document.id)

    after = await repo.get_document(pool, document.id)
    chunks = await repo.list_chunks(pool, document.id)
    assert outcome == "indexed"
    assert after.index_status == "indexed" and after.indexed_model == "fake/hash-embed"
    assert after.indexed_at is not None and after.index_detail is None
    assert {p.chunk_id for p in await stored(store, pool, document.id)} == {
        c.id for c in chunks
    }


async def test_the_search_text_is_what_gets_embedded_not_the_bare_text(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("Cash was strong."))

    await indexer.index_document(document.id)

    [chunk] = await repo.list_chunks(pool, document.id)
    assert embedder.calls == [[chunk.search_text]]
    assert "Liquidity" in chunk.search_text and chunk.search_text != chunk.text


async def test_identical_chunk_texts_are_embedded_once(
    pool, service, indexer, embedder
):
    def build(d):
        for _ in range(2):  # the same words under the same heading, as two chunks
            d.add_heading("Risk", 1)
            d.add_paragraph("Same words here.")

    document = await parsed(service, pool, mf.docx_bytes(build))
    assert len(await repo.list_chunks(pool, document.id)) == 2

    await indexer.index_document(document.id)

    assert embedder.texts_embedded == 1


async def test_a_document_that_is_not_pending_is_skipped(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("Some text."))
    await indexer.index_document(document.id)
    calls = embedder.texts_embedded

    assert await indexer.index_document(document.id) == "skipped"
    assert embedder.texts_embedded == calls


async def test_an_unparsed_document_is_not_indexed(pool, indexer):
    workspace = await repo.default_workspace_id(pool)
    document = await repo.create_document(
        pool, workspace_id=workspace, entity_id=None, filename="x.docx",
        kind="docx", sha256="a" * 64, size_bytes=1,
    )  # fmt: skip

    assert await indexer.index_document(document.id) == "skipped"


# --- re-parsing, reuse and stale points -----------------------------------------------------


async def test_a_reindex_reuses_vectors_of_unchanged_text(
    pool, service, indexer, embedder, store
):
    document = await parsed(
        service, pool, a_docx("The revolver matures in March 2027.")
    )
    await indexer.index_document(document.id)
    first = embedder.texts_embedded

    # a re-parse gives the chunks NEW ids but the same text
    await service.reparse(document.id)
    await wait_final(pool, document.id)
    assert await indexer.index_document(document.id) == "indexed"

    assert embedder.texts_embedded == first  # nothing was embedded again
    chunks = await repo.list_chunks(pool, document.id)
    assert {p.chunk_id for p in await stored(store, pool, document.id)} == {
        c.id for c in chunks
    }


async def test_points_of_chunks_that_no_longer_exist_are_removed(
    pool, service, indexer, store
):
    document = await parsed(service, pool, a_docx("First version of the text."))
    await indexer.index_document(document.id)
    old_ids = {p.chunk_id for p in await stored(store, pool, document.id)}

    await service.reparse(document.id)
    await wait_final(pool, document.id)
    await indexer.index_document(document.id)

    now = {p.chunk_id for p in await stored(store, pool, document.id)}
    assert now and not (now & old_ids)  # only the new chunk ids remain


async def test_assigning_an_entity_re_embeds_and_scopes_the_vectors(
    pool, service, indexer, embedder, store
):
    workspace = await repo.default_workspace_id(pool)
    acme = await repo.create_entity(pool, workspace, "Acme")
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))
    await indexer.index_document(document.id)
    before = embedder.texts_embedded

    await service.assign_entity(
        document.id, acme.id
    )  # rebuilds chunks; the search text now names Acme
    assert (await repo.get_document(pool, document.id)).index_status == "pending"
    await indexer.index_document(document.id)

    assert (
        embedder.texts_embedded == before + 1
    )  # the search text changed, so a new vector
    in_acme = await indexer.search(workspace, "revolver", entity_ids=[acme.id])
    elsewhere = await indexer.search(workspace, "revolver", entity_ids=[])
    assert [h.chunk.document_id for h in in_acme] == [document.id]
    assert elsewhere == []


# --- failures -----------------------------------------------------------------------------


async def test_a_failed_embedding_marks_the_document_failed_with_a_plain_reason(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))
    embedder.error = EmbeddingError("unavailable", "HTTP 503 secret-detail")

    outcome = await indexer.index_document(document.id)

    after = await repo.get_document(pool, document.id)
    assert outcome == "failed" and after.index_status == "failed"
    assert "retry later" in (after.index_detail or "")
    assert "secret-detail" not in (after.index_detail or "")
    # the document is still fully usable by full text
    assert await repo.search_chunks_text(pool, document.workspace_id, "revolver")


async def test_a_retry_after_a_failure_indexes_the_document(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))
    embedder.error = EmbeddingError("unavailable", "down")
    await indexer.index_document(document.id)

    embedder.error = None
    assert await indexer.reindex(document.id) is True
    done = await wait_index(pool, document.id, {"indexed", "failed"})

    assert done.index_status == "indexed" and done.index_detail is None


async def test_an_unexpected_error_is_recorded_without_leaking_it(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("Some text."))
    embedder.error = RuntimeError("connection string postgres://user:pw@host")

    outcome = await indexer.index_document(document.id)

    after = await repo.get_document(pool, document.id)
    assert outcome == "failed"
    assert after.index_detail == "unexpected error while indexing"


async def test_a_document_with_too_many_chunks_is_refused_before_any_embedding(
    pool, service, settings, embedder, store
):
    tight = settings.model_copy(update={"embedding_max_chunks_per_document": 1})
    ix = Indexer(pool, tight, embedder, store)

    def build(d):
        for i in range(3):
            d.add_heading(f"Section {i}", 1)
            d.add_paragraph(f"Paragraph {i} with some words.")

    document = await parsed(service, pool, mf.docx_bytes(build))

    assert await ix.index_document(document.id) == "failed"
    assert embedder.calls == []
    assert (
        "too many chunks" in (await repo.get_document(pool, document.id)).index_detail
    )


async def test_overlong_text_is_cut_for_embedding_and_the_cut_is_reported(
    pool, service, settings, embedder, store
):
    short = settings.model_copy(update={"embedding_max_chars": 60})
    ix = Indexer(pool, short, embedder, store)
    document = await parsed(service, pool, a_docx("Words " * 40))

    await ix.index_document(document.id)

    after = await repo.get_document(pool, document.id)
    assert after.index_status == "indexed" and "cut" in (after.index_detail or "")
    assert all(len(t) <= 60 for batch in embedder.calls for t in batch)


# --- concurrency and races ------------------------------------------------------------------


async def test_two_workers_never_index_the_same_document_twice(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))

    outcomes = await asyncio.gather(
        indexer.index_document(document.id), indexer.index_document(document.id)
    )

    assert sorted(outcomes) == ["indexed", "skipped"]
    assert embedder.texts_embedded == 1


async def test_a_document_changed_while_it_is_being_indexed_stays_pending(
    pool, service, indexer, embedder, settings
):
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))
    embedder.gate = asyncio.Event()
    running = asyncio.ensure_future(indexer.index_document(document.id))
    await embedder.entered.wait()  # the worker is inside the embedding call

    await service.reparse(document.id)  # new chunks: the index is stale again
    await wait_final(pool, document.id)
    embedder.gate.set()

    assert await running == "stale"
    assert (await repo.get_document(pool, document.id)).index_status == "pending"


async def test_cancelling_an_indexing_job_puts_the_document_back_to_pending(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))
    embedder.gate = asyncio.Event()  # never opened
    running = asyncio.ensure_future(indexer.index_document(document.id))
    await embedder.entered.wait()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert (await repo.get_document(pool, document.id)).index_status == "pending"


async def test_startup_recovery_resets_documents_left_indexing(pool, service):
    document = await parsed(service, pool, a_docx("Some text."))
    assert await repo.claim_for_indexing(pool, document.id)

    assert await repo.recover_interrupted_indexing(pool) == 1

    assert (await repo.get_document(pool, document.id)).index_status == "pending"


async def test_schedule_pending_indexes_everything_waiting(pool, service, indexer):
    first = await parsed(service, pool, a_docx("First document text."), name="a.docx")
    second = await parsed(service, pool, a_docx("Second document text."), name="b.docx")

    assert await indexer.schedule_pending() == 2

    for document in (first, second):
        assert (
            await wait_index(pool, document.id, {"indexed", "failed"})
        ).index_status == "indexed"


# --- searching ----------------------------------------------------------------------------


async def test_vector_search_finds_the_chunk_that_shares_the_most_words(
    pool, service, indexer
):
    document = await parsed(
        service,
        pool,
        a_docx("The revolver matures in March 2027.", "Our offices are in Dublin."),
    )
    await indexer.index_document(document.id)

    hits = await indexer.search(document.workspace_id, "when does the revolver mature")

    assert hits and "revolver" in hits[0].chunk.text
    assert hits[0].filename == "report.docx" and 0 < hits[0].score <= 1


async def test_vector_search_stays_inside_the_workspace(pool, service, indexer):
    document = await parsed(
        service, pool, a_docx("The revolver matures in March 2027.")
    )
    await indexer.index_document(document.id)
    other = await repo.create_workspace(pool, "Elsewhere")

    assert await indexer.search(other.id, "revolver") == []
    assert await indexer.search(document.workspace_id, "revolver")


async def test_a_vector_hit_whose_chunk_is_gone_is_dropped_not_an_error(
    pool, service, indexer, store
):
    document = await parsed(
        service, pool, a_docx("The revolver matures in March 2027.")
    )
    await indexer.index_document(document.id)
    await repo.save_chunks(
        pool,
        document.id,
        __import__("backend.app.chunking.models", fromlist=["ChunkSet"]).ChunkSet(),
    )

    assert await indexer.search(document.workspace_id, "revolver") == []


async def test_a_failed_query_embedding_raises_an_embedding_error(
    pool, service, indexer, embedder
):
    document = await parsed(service, pool, a_docx("The revolver matures in 2027."))
    await indexer.index_document(document.id)
    embedder.error = EmbeddingError("unavailable", "down")

    with pytest.raises(EmbeddingError):
        await indexer.search(document.workspace_id, "revolver")


async def test_search_results_can_feed_parent_expansion(pool, service, indexer):
    from backend.app.chunking.expand import expand_to_parents

    document = await parsed(
        service,
        pool,
        a_docx(
            "The revolver matures in March 2027.", "The revolver margin is 2 percent."
        ),
    )
    await indexer.index_document(document.id)

    hits = await indexer.search(document.workspace_id, "revolver")
    views = await expand_to_parents(pool, hits, max_parents=5)

    assert len(views) == 1 and "margin" in views[0].text


# --- HTTP ---------------------------------------------------------------------------------


@pytest.fixture
async def client(pool, settings, indexer) -> AsyncIterator[httpx.AsyncClient]:
    svc = DocumentService(pool, settings, indexer=indexer)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_documents] = lambda: svc
    app.dependency_overrides[get_indexer] = lambda: indexer
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await svc.close()


async def upload_over_http(client, data: bytes, name="report.docx"):
    response = await client.post(
        "/documents", files={"file": (name, data, "application/octet-stream")}
    )
    assert response.status_code == 201
    return response.json()


async def wait_http_indexed(client, document_id, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        body = (await client.get(f"/documents/{document_id}")).json()
        if body["index_status"] in {"indexed", "failed"}:
            return body
        await asyncio.sleep(0.05)
    raise AssertionError("never indexed")


async def test_uploading_a_document_indexes_it_in_the_background(client, pool):
    body = await upload_over_http(client, a_docx("The revolver matures in March 2027."))

    done = await wait_http_indexed(client, body["id"])

    assert done["status"] in {"parsed", "needs_review"}
    assert (
        done["index_status"] == "indexed" and done["indexed_model"] == "fake/hash-embed"
    )


async def test_the_search_route_does_text_and_vector_search(client):
    body = await upload_over_http(client, a_docx("The revolver matures in March 2027."))
    await wait_http_indexed(client, body["id"])
    workspace = body["workspace_id"]

    by_text = await client.get(
        f"/workspaces/{workspace}/search", params={"q": "revolver", "mode": "text"}
    )
    by_vector = await client.get(
        f"/workspaces/{workspace}/search", params={"q": "revolver", "mode": "vector"}
    )

    for response in (by_text, by_vector):
        assert response.status_code == 200
        [hit] = response.json()
        assert hit["filename"] == "report.docx" and "revolver" in hit["text"]
        assert hit["section_path"] == ["Liquidity"]


async def test_the_search_route_validates_its_input(client):
    workspace = (await client.get("/workspaces")).json()[0]["id"]

    assert (
        await client.get(f"/workspaces/{workspace}/search", params={"q": ""})
    ).status_code == 422
    assert (
        await client.get(
            f"/workspaces/{workspace}/search", params={"q": "x", "mode": "magic"}
        )
    ).status_code == 422
    assert (
        await client.get(
            f"/workspaces/{workspace}/search", params={"q": "x", "limit": 999}
        )
    ).status_code == 422
    unknown = "00000000-0000-4000-8000-000000000000"
    assert (
        await client.get(f"/workspaces/{unknown}/search", params={"q": "x"})
    ).status_code == 404


async def test_vector_search_reports_503_when_embedding_is_down(client, embedder):
    body = await upload_over_http(client, a_docx("The revolver matures in 2027."))
    await wait_http_indexed(client, body["id"])
    embedder.error = EmbeddingError("unavailable", "down")

    response = await client.get(
        f"/workspaces/{body['workspace_id']}/search",
        params={"q": "revolver", "mode": "vector"},
    )

    assert response.status_code == 503
    assert "down" not in response.text


async def test_the_index_route_retries_a_failed_document(client, embedder):
    embedder.error = EmbeddingError("unavailable", "down")
    body = await upload_over_http(client, a_docx("The revolver matures in 2027."))
    failed = await wait_http_indexed(client, body["id"])
    assert failed["index_status"] == "failed"

    embedder.error = None
    response = await client.post(f"/documents/{body['id']}/index")

    assert response.status_code == 202
    assert (await wait_http_indexed(client, body["id"]))["index_status"] == "indexed"


async def test_the_index_route_refuses_unknown_and_unparsed_documents(client):
    unknown = "00000000-0000-4000-8000-000000000000"

    assert (await client.post(f"/documents/{unknown}/index")).status_code == 404
