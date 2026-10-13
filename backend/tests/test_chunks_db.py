"""Chunks in the database: built after a parse, swapped atomically, rebuilt on entity change, and
searchable by full text inside a workspace. Each test uses a throwaway schema (conftest) and
skips without a database."""

import asyncio
import io
from collections.abc import AsyncIterator
from uuid import uuid4

import asyncpg
import httpx
import pytest

from backend.app.api_documents import get_documents
from backend.app.chunking.models import Chunk, ChunkSet, DocContext, Section
from backend.app.chunking.service import block_from_record, rechunk
from backend.app.chunking.splitter import build_chunks
from backend.app.config import get_settings
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import BlockRecord
from backend.app.documents_service import DocumentService
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import Block, canonical_text
from backend.app.main import app
from backend.tests import make_files as mf
from backend.tests.fakes import make_settings

FINAL = {"parsed", "needs_review", "needs_ocr", "failed"}


@pytest.fixture
def settings(tmp_path):
    return make_settings().model_copy(update={"upload_dir": str(tmp_path / "uploads")})


@pytest.fixture
async def service(pool, settings):
    svc = DocumentService(pool, settings)
    yield svc
    await svc.close()


def reader(data: bytes):
    buffer = io.BytesIO(data)

    async def read(size: int) -> bytes:
        return buffer.read(size)

    return read


def a_docx(*paragraphs: str) -> bytes:
    def build(d):
        d.add_heading("Liquidity", 1)
        for text in paragraphs:
            d.add_paragraph(text)

    return mf.docx_bytes(build)


async def wait_final(pool, document_id, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        document = await repo.get_document(pool, document_id)
        assert document is not None
        if document.status in FINAL:
            return document
        await asyncio.sleep(0.05)
    raise AssertionError("the document never finished parsing")


async def upload_and_parse(service, pool, data: bytes, name="report.docx", **kwargs):
    result = await service.upload(
        workspace_id=kwargs.get("workspace_id"),
        entity_id=kwargs.get("entity_id"),
        filename=name,
        read=reader(data),
    )
    return await wait_final(pool, result.document.id)


def stored_blocks_text(blocks) -> str:
    return "\n\n".join(b.text for b in blocks)


# --- stored blocks back into parser blocks ------------------------------------------------


def record(**overrides) -> BlockRecord:
    base: dict = {
        "id": uuid4(),
        "document_id": uuid4(),
        "seq": 0,
        "type": "paragraph",
        "text": "Hello.",
        "level": None,
        "location": {"section_path": ["A"], "page": 2},
        "data": None,
        "confidence": 0.8,
        "flags": ["math_notation"],
        "char_start": 0,
        "char_end": 6,
    }
    base.update(overrides)
    return BlockRecord(**base)


def test_a_stored_paragraph_comes_back_with_its_location_confidence_and_flags():
    block = block_from_record(record())

    assert block.type == "paragraph" and block.text == "Hello."
    assert block.location.section_path == ["A"] and block.location.page == 2
    assert block.confidence == 0.8 and block.flags == ["math_notation"]
    assert block.table is None


def test_a_stored_table_comes_back_as_structured_cells():
    data = {
        "rows": [["Item", "FY24"], ["Revenue", "100"]],
        "header_rows": 1,
        "units": "in millions",
        "has_merged_cells": False,
    }

    block = block_from_record(record(type="table", text="Item | FY24", data=data))

    assert block.table is not None
    assert (
        block.table.rows[1] == ["Revenue", "100"] and block.table.units == "in millions"
    )


def test_a_stored_figure_keeps_its_labels():
    block = block_from_record(
        record(type="figure", text="Chart", data={"labels": ["2023", "2024"]})
    )

    assert block.labels == ["2023", "2024"]


# --- built when a document is parsed ------------------------------------------------------


async def test_a_parsed_document_has_chunks_that_are_exact_slices_of_its_blocks(
    pool, service
):
    document = await upload_and_parse(
        service, pool, a_docx("Cash was strong in 2024.", "Debt fell by ten percent.")
    )

    blocks = await repo.list_blocks(pool, document.id)
    chunks = await repo.list_chunks(pool, document.id)
    canon = stored_blocks_text(blocks)

    assert document.status == "parsed" and chunks
    for c in chunks:
        assert (
            c.text == canon[c.char_start : c.char_end]
        )  # the citation text is verbatim
        assert c.search_text.endswith("\n" + c.text)
        assert "report.docx" in c.search_text.split("\n")[0]
        assert c.section_path == ["Liquidity"]


async def test_chunks_have_a_parent_section_holding_their_siblings(pool, service):
    document = await upload_and_parse(service, pool, a_docx("One thing.", "Another."))

    sections = await repo.list_sections(pool, document.id)
    chunks = await repo.list_chunks(pool, document.id)
    siblings = await repo.get_section_chunks(pool, chunks[0].section_id)

    assert [s.path for s in sections] == [["Liquidity"]]
    assert [c.id for c in siblings] == [c.id for c in chunks]


async def test_a_reparse_replaces_the_chunks(pool, service):
    document = await upload_and_parse(service, pool, a_docx("Some text here."))
    before = {c.id for c in await repo.list_chunks(pool, document.id)}

    await service.reparse(document.id)
    await wait_final(pool, document.id)
    after = await repo.list_chunks(pool, document.id)

    assert len(after) == len(before) and not ({c.id for c in after} & before)


async def test_a_failed_chunk_save_leaves_the_old_blocks_and_chunks_untouched(pool):
    workspace = await repo.default_workspace_id(pool)
    doc = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.txt",
        kind="txt",
        sha256="a" * 64,
        size_bytes=5,
    )
    builder = BlockBuilder()
    builder.heading("Intro", 1)
    builder.paragraph("Original words.")
    good = build_chunks(builder.blocks, DocContext(label="a.txt"))
    await repo.save_parse_result(
        pool, doc.id, parser="t", status="parsed", detail=None, page_count=None,
        char_count=10, report={}, blocks=builder.blocks, chunks=good,
    )  # fmt: skip
    broken = ChunkSet(
        parents=[Section(ordinal=0, path=[], char_start=0, char_end=1, token_count=1)],
        chunks=[
            Chunk(
                ordinal=0, parent_ordinal=0, kind="prose", text="", search_text="\n",
                section_path=[], block_seqs=[0], char_start=0, char_end=1, token_count=1,
            )
        ],
    )  # fmt: skip

    with pytest.raises(asyncpg.PostgresError):
        await repo.save_parse_result(
            pool, doc.id, parser="t", status="parsed", detail=None, page_count=None,
            char_count=3, report={}, blocks=[Block(type="paragraph", text="New")],
            chunks=broken,
        )  # fmt: skip

    assert [b.text for b in await repo.list_blocks(pool, doc.id)] == [
        "Intro",
        "Original words.",
    ]
    assert [c.text for c in await repo.list_chunks(pool, doc.id)] == ["Original words."]


async def test_deleting_a_document_deletes_its_chunks_and_sections(pool, service):
    document = await upload_and_parse(service, pool, a_docx("Some text."))

    await pool.execute("DELETE FROM documents WHERE id = $1", document.id)

    assert await pool.fetchval("SELECT count(*) FROM document_chunks") == 0
    assert await pool.fetchval("SELECT count(*) FROM document_sections") == 0


async def test_a_document_with_no_content_has_no_chunks(pool, service):
    document = await upload_and_parse(service, pool, b"   \n  ", name="empty.txt")

    assert await repo.count_chunks(pool, document.id) == 0


# --- entity assignment rebuilds the search text -------------------------------------------


async def test_assigning_an_entity_puts_its_name_in_the_search_text(pool, service):
    workspace = await repo.default_workspace_id(pool)
    entity = await repo.create_entity(pool, workspace, "Acme")
    document = await upload_and_parse(service, pool, a_docx("Cash was strong."))
    before = (await repo.list_chunks(pool, document.id))[0]
    assert "Acme" not in before.search_text

    await service.assign_entity(document.id, entity.id)
    after = (await repo.list_chunks(pool, document.id))[0]

    assert "(Acme)" in after.search_text.split("\n")[0]
    assert after.text == before.text  # the citation text did not change
    assert after.content_hash != before.content_hash  # so it would be embedded again


async def test_detaching_an_entity_takes_its_name_out_again(pool, service):
    workspace = await repo.default_workspace_id(pool)
    entity = await repo.create_entity(pool, workspace, "Acme")
    document = await upload_and_parse(
        service, pool, a_docx("Cash."), entity_id=entity.id
    )

    await service.assign_entity(document.id, None)

    assert "Acme" not in (await repo.list_chunks(pool, document.id))[0].search_text


async def test_rechunk_of_an_unknown_document_is_none(pool, settings):
    assert await rechunk(pool, settings, uuid4()) is None


async def test_rechunk_gives_the_same_chunks_as_the_first_parse(
    pool, service, settings
):
    document = await upload_and_parse(
        service, pool, a_docx("First paragraph.", "Second paragraph.")
    )
    before = [
        (c.text, c.char_start, c.char_end)
        for c in await repo.list_chunks(pool, document.id)
    ]

    await rechunk(pool, settings, document.id)
    after = [
        (c.text, c.char_start, c.char_end)
        for c in await repo.list_chunks(pool, document.id)
    ]

    assert after == before


# --- full-text search ---------------------------------------------------------------------


async def test_search_finds_a_chunk_by_its_words_and_by_its_section_heading(
    pool, service
):
    document = await upload_and_parse(
        service, pool, a_docx("The revolver matures in March 2027.")
    )
    workspace = document.workspace_id

    by_text = await repo.search_chunks_text(pool, workspace, "revolver matures")
    by_heading = await repo.search_chunks_text(pool, workspace, "Liquidity revolver")

    assert [h.chunk.document_id for h in by_text] == [document.id]
    assert by_text[0].filename == "report.docx" and by_text[0].score > 0
    assert by_heading  # the heading is in search_text, not in the chunk's own text


async def test_search_only_returns_chunks_from_the_asked_workspace(pool, service):
    document = await upload_and_parse(
        service, pool, a_docx("The revolver matures soon.")
    )
    other = await repo.create_workspace(pool, "Elsewhere")

    hits = await repo.search_chunks_text(pool, other.id, "revolver")

    assert hits == []
    assert await repo.search_chunks_text(pool, document.workspace_id, "revolver")


async def test_search_can_be_limited_to_some_entities(pool, service):
    workspace = await repo.default_workspace_id(pool)
    acme = await repo.create_entity(pool, workspace, "Acme")
    beta = await repo.create_entity(pool, workspace, "Beta")
    await upload_and_parse(
        service, pool, a_docx("Acme revolver terms."), name="a.docx", entity_id=acme.id
    )
    await upload_and_parse(
        service, pool, a_docx("Beta revolver terms."), name="b.docx", entity_id=beta.id
    )

    only_acme = await repo.search_chunks_text(
        pool, workspace, "revolver", entity_ids=[acme.id]
    )
    both = await repo.search_chunks_text(pool, workspace, "revolver")

    assert {h.filename for h in only_acme} == {"a.docx"}
    assert {h.filename for h in both} == {"a.docx", "b.docx"}


@pytest.mark.parametrize(
    "query", ["", "   ", '"unclosed', "a OR", "-", "(((", "x' OR '1'='1"]
)
async def test_odd_search_input_never_raises(pool, service, query):
    document = await upload_and_parse(service, pool, a_docx("Some words."))

    hits = await repo.search_chunks_text(pool, document.workspace_id, query)

    assert isinstance(hits, list)


async def test_search_is_the_same_whatever_the_case(pool, service):
    document = await upload_and_parse(service, pool, a_docx("EBITDA improved."))

    hits = await repo.search_chunks_text(pool, document.workspace_id, "ebitda")

    assert len(hits) == 1


# --- over HTTP ----------------------------------------------------------------------------


@pytest.fixture
async def client(pool, settings, service) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_documents] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_the_chunks_route_lists_chunks_and_the_document_reports_the_count(
    client, pool, service
):
    document = await upload_and_parse(service, pool, a_docx("Cash was strong."))

    chunks = (await client.get(f"/documents/{document.id}/chunks")).json()
    view = (await client.get(f"/documents/{document.id}")).json()

    assert [c["text"] for c in chunks] == ["Cash was strong."]
    assert view["chunk_count"] == 1


async def test_the_chunks_route_404s_for_an_unknown_document(client):
    assert (await client.get(f"/documents/{uuid4()}/chunks")).status_code == 404


async def test_assigning_an_entity_over_http_rebuilds_the_chunks(client, pool, service):
    workspace = await repo.default_workspace_id(pool)
    entity = await repo.create_entity(pool, workspace, "Acme")
    document = await upload_and_parse(service, pool, a_docx("Cash was strong."))

    response = await client.patch(
        f"/documents/{document.id}", json={"entity_id": str(entity.id)}
    )

    assert response.status_code == 200
    chunks = (await client.get(f"/documents/{document.id}/chunks")).json()
    assert "(Acme)" in chunks[0]["search_text"]


def test_canonical_text_helper_matches_the_stored_join():
    blocks = [Block(type="paragraph", text="A"), Block(type="paragraph", text="B")]

    assert canonical_text(blocks) == "A\n\nB"
