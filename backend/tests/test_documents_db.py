"""Documents against a real database: the repository, the upload + background parse service, and
the failure paths. Each test runs in its own throwaway schema (see conftest) and skips without a DB."""

import asyncio
import io
import threading
import time
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from backend.app import documents_service
from backend.app.db import repository as repo
from backend.app.documents_service import DocumentService, ParseBusyError
from backend.app.ingest.intake import IntakeError
from backend.app.ingest.models import Block, Location, TableData, canonical_text
from backend.app.ingest.storage import path_for
from backend.tests import make_files as mf
from backend.tests.fakes import make_settings
from backend.tests.test_parse_pdf import two_pages_text_then_scan

SHA = "a" * 64


def reader(data: bytes):
    buffer = io.BytesIO(data)

    async def read(size: int) -> bytes:
        return buffer.read(size)

    return read


@pytest.fixture
def settings(tmp_path):
    return make_settings().model_copy(update={"upload_dir": str(tmp_path / "uploads")})


@pytest.fixture
async def service(pool, settings):
    svc = DocumentService(pool, settings)
    yield svc
    await svc.close()


@pytest.fixture
async def workspace(pool):
    return await repo.default_workspace_id(pool)


async def wait_for_status(pool, document_id, wanted, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        document = await repo.get_document(pool, document_id)
        assert document is not None
        if document.status in wanted:
            return document
        await asyncio.sleep(0.05)
    raise AssertionError(f"document stayed '{document.status}', wanted one of {wanted}")


async def upload(service, data: bytes, name: str, **kwargs):
    return await service.upload(
        workspace_id=kwargs.get("workspace_id"),
        entity_id=kwargs.get("entity_id"),
        filename=name,
        read=reader(data),
    )


def files_on_disk(settings):
    root = Path(settings.upload_dir)
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def a_block(text="text", kind="paragraph", **kwargs) -> Block:
    return Block(type=kind, text=text, location=Location(page=1), **kwargs)


# =========================================================================================
# repository
# =========================================================================================


async def test_a_document_is_recorded_as_uploaded(pool, workspace):
    document = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="r.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=10,
    )

    assert (
        document.status == "uploaded"
        and document.entity_id is None
        and document.parsed_at is None
    )
    assert (
        document.report == {}
        and (await repo.get_document(pool, document.id)) == document
    )


async def test_the_same_bytes_in_one_workspace_are_one_document(pool, workspace):
    first = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=10,
    )

    with pytest.raises(repo.DuplicateDocumentError) as caught:
        await repo.create_document(
            pool,
            workspace_id=workspace,
            entity_id=None,
            filename="copy.pdf",
            kind="pdf",
            sha256=SHA,
            size_bytes=10,
        )

    assert (
        caught.value.existing.id == first.id
        and caught.value.existing.filename == "a.pdf"
    )


async def test_the_same_bytes_in_another_workspace_are_a_separate_document(
    pool, workspace
):
    other = await repo.create_workspace(pool, "other")
    one = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=10,
    )

    two = await repo.create_document(
        pool,
        workspace_id=other.id,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=10,
    )

    assert one.id != two.id


async def test_unknown_workspace_and_foreign_entity_are_refused(pool, workspace):
    other = await repo.create_workspace(pool, "other")
    foreign = await repo.create_entity(pool, other.id, "Visa")

    with pytest.raises(repo.WorkspaceNotFoundError):
        await repo.create_document(
            pool,
            workspace_id=uuid4(),
            entity_id=None,
            filename="a.pdf",
            kind="pdf",
            sha256=SHA,
            size_bytes=1,
        )
    with pytest.raises(repo.EntityNotInWorkspaceError):
        await repo.create_document(
            pool,
            workspace_id=workspace,
            entity_id=foreign.id,
            filename="a.pdf",
            kind="pdf",
            sha256=SHA,
            size_bytes=1,
        )
    with pytest.raises(repo.EntityNotInWorkspaceError):
        await repo.create_document(
            pool,
            workspace_id=workspace,
            entity_id=uuid4(),
            filename="a.pdf",
            kind="pdf",
            sha256=SHA,
            size_bytes=1,
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("sha256", "not-a-hash"),
        ("sha256", "A" * 64),
        ("size_bytes", 0),
        ("kind", "exe"),
        ("filename", ""),
        ("status", "weird"),
    ],
)
async def test_the_database_rejects_bad_document_rows(pool, workspace, column, value):
    good = {
        "workspace_id": workspace,
        "filename": "a.pdf",
        "kind": "pdf",
        "sha256": SHA,
        "size_bytes": 5,
        "status": "uploaded",
    }
    good[column] = value
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO documents (workspace_id, filename, kind, sha256, size_bytes, status) VALUES ($1, $2, $3, $4, $5, $6)",
                good["workspace_id"], good["filename"], good["kind"], good["sha256"], good["size_bytes"], good["status"],
            )  # fmt: skip


async def test_the_database_itself_enforces_that_an_entity_belongs_to_the_documents_workspace(
    pool, workspace
):
    other = await repo.create_workspace(pool, "other")
    foreign = await repo.create_entity(pool, other.id, "Visa")
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await conn.execute(
                "INSERT INTO documents (workspace_id, entity_id, filename, kind, sha256, size_bytes) VALUES ($1, $2, 'a.pdf', 'pdf', $3, 5)",
                workspace, foreign.id, SHA,
            )  # fmt: skip


async def test_deleting_an_entity_unassigns_its_documents_but_keeps_them(
    pool, workspace
):
    entity = await repo.create_entity(pool, workspace, "Visa")
    document = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=entity.id,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM entities WHERE id = $1", entity.id)

    kept = await repo.get_document(pool, document.id)
    assert (
        kept is not None and kept.entity_id is None and kept.workspace_id == workspace
    )


async def test_deleting_a_workspace_removes_its_documents_and_blocks(pool):
    doomed = await repo.create_workspace(pool, "doomed")
    document = await repo.create_document(
        pool,
        workspace_id=doomed.id,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )
    await repo.save_parse_result(
        pool,
        document.id,
        parser="t",
        status="parsed",
        detail=None,
        page_count=1,
        char_count=4,
        report={},
        blocks=[a_block("text")],
    )

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM workspaces WHERE id = $1", doomed.id)
        counts = await conn.fetchrow(
            "SELECT (SELECT count(*) FROM documents) AS d, (SELECT count(*) FROM document_blocks) AS b"
        )

    assert counts is not None and (counts["d"], counts["b"]) == (0, 0)


async def test_assign_and_unassign_an_entity(pool, workspace):
    entity = await repo.create_entity(pool, workspace, "Visa")
    document = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )

    await repo.assign_entity(pool, document.id, entity.id)
    assert (await repo.get_document(pool, document.id)).entity_id == entity.id  # type: ignore[union-attr]
    await repo.assign_entity(pool, document.id, None)
    assert (await repo.get_document(pool, document.id)).entity_id is None  # type: ignore[union-attr]


async def test_assigning_a_foreign_entity_or_to_a_missing_document_fails(
    pool, workspace
):
    other = await repo.create_workspace(pool, "other")
    foreign = await repo.create_entity(pool, other.id, "Visa")
    document = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )

    with pytest.raises(repo.EntityNotInWorkspaceError):
        await repo.assign_entity(pool, document.id, foreign.id)
    with pytest.raises(LookupError):
        await repo.assign_entity(pool, uuid4(), None)


async def test_listing_documents_filters_by_entity_and_pages(pool, workspace):
    entity = await repo.create_entity(pool, workspace, "Visa")
    for i in range(3):
        await repo.create_document(
            pool,
            workspace_id=workspace,
            entity_id=entity.id if i < 2 else None,
            filename=f"{i}.pdf",
            kind="pdf",
            sha256=f"{i}" * 64,
            size_bytes=1,
        )

    assert len(await repo.list_documents(pool, workspace)) == 3
    assert len(await repo.list_documents(pool, workspace, entity_id=entity.id)) == 2
    assert len(await repo.list_documents(pool, workspace, limit=2)) == 2
    assert len(await repo.list_documents(pool, workspace, limit=2, offset=2)) == 1


# --- claiming and recovery ---------------------------------------------------------------


async def test_only_one_worker_can_claim_a_document(pool, workspace):
    document = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )

    claims = await asyncio.gather(
        *(repo.claim_for_parsing(pool, document.id) for _ in range(6))
    )

    assert sum(c is not None for c in claims) == 1
    assert (await repo.get_document(pool, document.id)).status == "parsing"  # type: ignore[union-attr]


async def test_claiming_a_missing_document_returns_nothing(pool):
    assert await repo.claim_for_parsing(pool, uuid4()) is None


async def test_interrupted_documents_are_marked_failed_after_a_restart(pool, workspace):
    states = {}
    for i, status in enumerate(["uploaded", "parsing", "parsed", "needs_review"]):
        document = await repo.create_document(
            pool,
            workspace_id=workspace,
            entity_id=None,
            filename=f"{i}.pdf",
            kind="pdf",
            sha256=f"{i}" * 64,
            size_bytes=1,
        )
        await repo.set_document_status(pool, document.id, status)
        states[status] = document.id

    assert await repo.recover_interrupted(pool) == 2

    after = {s: (await repo.get_document(pool, i)).status for s, i in states.items()}  # type: ignore[union-attr]
    assert after == {
        "uploaded": "failed",
        "parsing": "failed",
        "parsed": "parsed",
        "needs_review": "needs_review",
    }
    failed = await repo.get_document(pool, states["parsing"])
    assert failed is not None and "re-parse" in (failed.status_detail or "")


# --- parse results ----------------------------------------------------------------------


def sample_blocks() -> list[Block]:
    table = Block(
        type="table",
        text="Region | Sales\nEU | 5",
        location=Location(page=2, bbox=(1.0, 2.0, 3.0, 4.0), section_path=["Results"]),
        table=TableData(
            rows=[["Region", "Sales"], ["EU", "5"]], header_rows=1, units="in millions"
        ),
        confidence=0.9,
        flags=["stitched_across_pages:2-3"],
    )
    figure = Block(
        type="figure",
        text="Figure 1\nFiscal year",
        labels=["Fiscal year"],
        location=Location(page=3),
        flags=["chart_data_not_extracted"],
        confidence=0.8,
    )
    return [
        Block(
            type="heading",
            text="Results",
            level=1,
            location=Location(page=1, section_path=["Results"]),
        ),
        a_block("Revenue grew."),
        table,
        figure,
        Block(type="list_item", text="- point", level=0, location=Location(page=3)),
    ]


async def save(pool, document, blocks, **overrides):
    args = {
        "parser": "test/1",
        "status": "parsed",
        "detail": None,
        "page_count": 3,
        "char_count": sum(len(b.text) for b in blocks),
        "report": {"flags": []},
        "blocks": blocks,
    }
    args.update(overrides)
    await repo.save_parse_result(pool, document.id, **args)


@pytest.fixture
async def document(pool, workspace):
    return await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )


async def test_blocks_round_trip_in_reading_order_with_structure_intact(pool, document):
    blocks = sample_blocks()
    await save(
        pool, document, blocks, report={"flags": ["x"], "block_counts": {"table": 1}}
    )

    stored = await repo.list_blocks(pool, document.id)

    assert [b.type for b in stored] == [
        "heading",
        "paragraph",
        "table",
        "figure",
        "list_item",
    ]
    assert [b.seq for b in stored] == [0, 1, 2, 3, 4]
    table = stored[2]
    assert table.data is not None and table.data["rows"] == [
        ["Region", "Sales"],
        ["EU", "5"],
    ]
    assert table.data["units"] == "in millions" and table.data["header_rows"] == 1
    assert table.location == {
        "page": 2,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "section_path": ["Results"],
    }
    assert (
        table.flags == ["stitched_across_pages:2-3"]
        and abs(table.confidence - 0.9) < 1e-6
    )
    assert (
        stored[3].data == {"labels": ["Fiscal year"]}
        and stored[0].data is None
        and stored[0].level == 1
    )
    refreshed = await repo.get_document(pool, document.id)
    assert refreshed is not None
    assert (
        refreshed.status,
        refreshed.parser,
        refreshed.page_count,
        refreshed.report["flags"],
    ) == ("parsed", "test/1", 3, ["x"])
    assert refreshed.parsed_at is not None and refreshed.char_count == sum(
        len(b.text) for b in blocks
    )


async def test_stored_offsets_point_into_the_documents_canonical_text(pool, document):
    blocks = sample_blocks()
    await save(pool, document, blocks)

    stored = await repo.list_blocks(pool, document.id)

    canon = canonical_text(blocks)
    assert "\n\n".join(b.text for b in stored) == canon
    for block in stored:
        assert canon[block.char_start : block.char_end] == block.text


async def test_saving_again_replaces_the_old_blocks_atomically(pool, document):
    await save(pool, document, sample_blocks())

    await save(
        pool,
        document,
        [a_block("only one now")],
        status="needs_review",
        detail="changed",
    )

    stored = await repo.list_blocks(pool, document.id)
    assert [b.text for b in stored] == ["only one now"] and stored[0].seq == 0
    assert (await repo.get_document(pool, document.id)).status == "needs_review"  # type: ignore[union-attr]


async def test_a_failed_save_leaves_the_previous_blocks_untouched(pool, document):
    await save(pool, document, [a_block("keep me")])
    broken = Block.model_construct(
        type="paragraph",
        text="x",
        location=Location(page=1),
        level=None,
        table=None,
        labels=[],
        confidence=2.0,
        flags=[],
    )  # violates the CHECK

    with pytest.raises(asyncpg.CheckViolationError):
        await save(pool, document, [a_block("new"), broken])

    assert [b.text for b in await repo.list_blocks(pool, document.id)] == [
        "keep me"
    ]  # rolled back


async def test_saving_for_a_missing_document_raises_and_stores_nothing(pool):
    with pytest.raises(LookupError):
        await repo.save_parse_result(
            pool,
            uuid4(),
            parser="t",
            status="parsed",
            detail=None,
            page_count=None,
            char_count=1,
            report={},
            blocks=[],
        )


async def test_blocks_can_be_filtered_and_paged(pool, document):
    await save(pool, document, sample_blocks())

    assert [
        b.type for b in await repo.list_blocks(pool, document.id, block_type="table")
    ] == ["table"]
    assert [b.seq for b in await repo.list_blocks(pool, document.id, limit=2)] == [0, 1]
    assert [
        b.seq for b in await repo.list_blocks(pool, document.id, limit=2, offset=3)
    ] == [3, 4]
    assert await repo.count_blocks(pool, document.id) == 5
    assert await repo.count_blocks(pool, document.id, "paragraph") == 1
    assert await repo.list_blocks(pool, uuid4()) == []


async def test_the_database_rejects_impossible_blocks(pool, document):
    await save(pool, document, [a_block("one")])
    async with pool.acquire() as conn:
        for bad_sql, args in [
            ("INSERT INTO document_blocks (document_id, seq, type, text, location, confidence, char_start, char_end) VALUES ($1, 5, 'paragraph', 'x', '{}', 1.5, 0, 1)", (document.id,)),
            ("INSERT INTO document_blocks (document_id, seq, type, text, location, confidence, char_start, char_end) VALUES ($1, 5, 'paragraph', 'x', '{}', 1.0, 9, 1)", (document.id,)),
            ("INSERT INTO document_blocks (document_id, seq, type, text, location, confidence, char_start, char_end) VALUES ($1, 5, 'banner', 'x', '{}', 1.0, 0, 1)", (document.id,)),
        ]:  # fmt: skip
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(bad_sql, *args)
        with pytest.raises(asyncpg.UniqueViolationError):  # seq 0 already exists
            await conn.execute(
                "INSERT INTO document_blocks (document_id, seq, type, text, location, confidence, char_start, char_end) VALUES ($1, 0, 'paragraph', 'x', '{}', 1.0, 0, 1)",
                document.id,
            )


# =========================================================================================
# service: upload and background parse
# =========================================================================================


async def test_an_upload_is_stored_by_hash_parsed_in_the_background_and_saved(
    pool, service, settings
):
    def build(d):
        d.add_heading("Annual report", 1)
        d.add_paragraph("Revenue grew strongly.")

    result = await upload(service, mf.docx_bytes(build), "../../evil name.docx")

    assert result.duplicate is False and result.document.status == "uploaded"
    assert result.document.filename == "evil name.docx"  # a label only
    assert path_for(Path(settings.upload_dir), result.document.sha256).exists()
    assert all("evil" not in str(p) for p in files_on_disk(settings))

    parsed = await wait_for_status(
        pool, result.document.id, {"parsed", "needs_review", "failed"}
    )

    assert (
        parsed.status == "parsed"
        and parsed.parser == "docx/python-docx"
        and parsed.status_detail is None
    )
    blocks = await repo.list_blocks(pool, parsed.id)
    assert [(b.type, b.text) for b in blocks] == [
        ("heading", "Annual report"),
        ("paragraph", "Revenue grew strongly."),
    ]
    assert blocks[1].location["section_path"] == ["Annual report"]
    assert parsed.report["block_counts"] == {"heading": 1, "paragraph": 1}
    assert parsed.char_count == sum(len(b.text) for b in blocks)


@pytest.mark.parametrize(
    ("name", "make"),
    [
        pytest.param(
            "a.pdf", lambda: mf.pdf_text("A real sentence in a PDF."), id="pdf"
        ),
        pytest.param("a.xlsx", mf.xlsx_bytes, id="xlsx"),
        pytest.param("a.pptx", mf.pptx_bytes, id="pptx"),
        pytest.param(
            "a.html", lambda: b"<html><body><p>Hello world</p></body></html>", id="html"
        ),
        pytest.param("a.csv", lambda: b"a,b\n1,2\n", id="csv"),
        pytest.param("a.md", lambda: b"# Title\n\nbody", id="md"),
        pytest.param("a.txt", lambda: b"plain words", id="txt"),
    ],
)
async def test_every_supported_kind_goes_from_upload_to_parsed(
    pool, service, name, make
):
    result = await upload(service, make(), name)

    done = await wait_for_status(
        pool, result.document.id, {"parsed", "needs_review", "needs_ocr", "failed"}
    )

    assert done.status == "parsed", done.status_detail
    assert await repo.count_blocks(pool, done.id) > 0


async def test_uploading_the_same_bytes_again_returns_the_existing_document_and_does_not_reparse(
    pool, service, monkeypatch
):
    calls = []
    original = documents_service.parse_file
    monkeypatch.setattr(
        documents_service,
        "parse_file",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    data = mf.docx_bytes()
    first = await upload(service, data, "a.docx")
    await wait_for_status(pool, first.document.id, {"parsed"})

    second = await upload(service, data, "renamed copy.docx")

    assert second.duplicate is True and second.document.id == first.document.id
    assert second.document.filename == "a.docx" and len(calls) == 1
    assert len(await repo.list_documents(pool, first.document.workspace_id)) == 1


async def test_the_same_file_in_two_workspaces_is_two_documents_sharing_one_file_on_disk(
    pool, service, settings
):
    other = await repo.create_workspace(pool, "other")
    data = mf.docx_bytes()

    one = await upload(service, data, "a.docx")
    two = await upload(service, data, "a.docx", workspace_id=other.id)

    assert one.document.id != two.document.id and not two.duplicate
    assert len(files_on_disk(settings)) == 1
    await wait_for_status(pool, one.document.id, {"parsed"})
    await wait_for_status(pool, two.document.id, {"parsed"})


async def test_a_request_for_an_unknown_workspace_or_foreign_entity_stores_nothing(
    pool, service, settings, workspace
):
    other = await repo.create_workspace(pool, "other")
    foreign = await repo.create_entity(pool, other.id, "Visa")

    with pytest.raises(repo.WorkspaceNotFoundError):
        await upload(service, mf.docx_bytes(), "a.docx", workspace_id=uuid4())
    with pytest.raises(repo.EntityNotInWorkspaceError):
        await upload(
            service,
            mf.docx_bytes(),
            "a.docx",
            workspace_id=workspace,
            entity_id=foreign.id,
        )

    assert (
        files_on_disk(settings) == []
        and await repo.list_documents(pool, workspace) == []
    )


async def test_a_refused_file_leaves_no_document_and_no_file(
    pool, service, settings, workspace
):
    macro = mf.add_to_zip(mf.docx_bytes(), {"word/vbaProject.bin": b"macro"})

    with pytest.raises(IntakeError) as caught:
        await upload(service, macro, "innocent.docx")

    assert caught.value.code == "macro_enabled"
    assert (
        files_on_disk(settings) == []
        and await repo.list_documents(pool, workspace) == []
    )


async def test_a_document_can_be_uploaded_directly_to_an_entity(
    pool, service, workspace
):
    entity = await repo.create_entity(pool, workspace, "Alphabet")

    result = await upload(service, mf.docx_bytes(), "a.docx", entity_id=entity.id)

    assert result.document.entity_id == entity.id
    await wait_for_status(pool, result.document.id, {"parsed"})


# --- statuses from real files ------------------------------------------------------------


async def test_a_scanned_pdf_ends_as_needs_ocr_with_no_blocks(pool, service):
    result = await upload(service, mf.pdf_scanned(), "scan.pdf")

    done = await wait_for_status(
        pool, result.document.id, {"needs_ocr", "parsed", "needs_review", "failed"}
    )

    assert done.status == "needs_ocr" and "OCR" in (done.status_detail or "")
    assert await repo.count_blocks(pool, done.id) == 0 and done.page_count == 1


async def test_a_mixed_pdf_keeps_its_text_and_is_marked_needs_review(pool, service):
    result = await upload(service, two_pages_text_then_scan(), "mixed.pdf")

    done = await wait_for_status(
        pool, result.document.id, {"needs_review", "parsed", "needs_ocr", "failed"}
    )

    assert done.status == "needs_review" and "page(s) 2" in (done.status_detail or "")
    assert await repo.count_blocks(pool, done.id) >= 1


async def test_a_table_whose_totals_do_not_add_up_makes_the_document_needs_review(
    pool, service
):
    def build(d):
        table = d.add_table(rows=4, cols=2)
        for r, (label, value) in enumerate(
            [("Item", "2024"), ("A", "10"), ("B", "20"), ("Total", "35")]
        ):
            table.cell(r, 0).text, table.cell(r, 1).text = label, value

    result = await upload(service, mf.docx_bytes(build), "bad-totals.docx")

    done = await wait_for_status(
        pool, result.document.id, {"needs_review", "parsed", "failed"}
    )

    assert done.status == "needs_review" and "do not add up" in (
        done.status_detail or ""
    )
    (table,) = await repo.list_blocks(pool, done.id, block_type="table")
    assert any(f.startswith("table_total_mismatch") for f in table.flags)


# --- failures never escape and never leak -------------------------------------------------


async def test_a_file_that_passes_intake_but_cannot_be_parsed_is_marked_failed_with_a_reason(
    pool, service
):
    broken = mf.zip_bytes(
        {"[Content_Types].xml": b"<Types/>", "word/document.xml": b"not xml at all"}
    )
    result = await upload(service, broken, "broken.docx")

    done = await wait_for_status(
        pool, result.document.id, {"failed", "parsed", "needs_review"}
    )

    assert (
        done.status == "failed"
        and done.status_detail == "The Word file could not be opened."
    )
    assert await repo.count_blocks(pool, done.id) == 0


async def test_an_unexpected_parser_crash_is_failed_without_leaking_internals(
    pool, service, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("secret internal path C:/very/private")

    monkeypatch.setattr(documents_service, "parse_file", explode)
    result = await upload(service, mf.docx_bytes(), "a.docx")

    done = await wait_for_status(pool, result.document.id, {"failed"})

    assert done.status_detail == "unexpected error while reading the file"
    assert "secret" not in repr(done)


async def test_a_parse_that_takes_too_long_is_failed(pool, settings, monkeypatch):
    slow_settings = settings.model_copy(update={"parse_timeout_s": 0.2})
    service = DocumentService(pool, slow_settings)
    monkeypatch.setattr(
        documents_service, "parse_file", lambda *a, **k: time.sleep(1.5)
    )
    try:
        result = await upload(service, mf.docx_bytes(), "a.docx")
        done = await wait_for_status(pool, result.document.id, {"failed"})
    finally:
        await service.close(timeout=3)

    assert "longer than" in (done.status_detail or "")


async def test_a_missing_stored_file_is_failed_not_crashed(pool, service, workspace):
    document = await repo.create_document(
        pool,
        workspace_id=workspace,
        entity_id=None,
        filename="a.pdf",
        kind="pdf",
        sha256=SHA,
        size_bytes=1,
    )

    await service.parse(document.id)

    done = await repo.get_document(pool, document.id)
    assert done is not None and (done.status, done.status_detail) == (
        "failed",
        "the stored file is missing",
    )


# --- re-parse, concurrency ---------------------------------------------------------------


async def test_reparse_replaces_the_blocks_without_duplicating_them(pool, service):
    result = await upload(
        service, mf.docx_bytes(lambda d: d.add_paragraph("one paragraph")), "a.docx"
    )
    await wait_for_status(pool, result.document.id, {"parsed"})

    claimed = await service.reparse(result.document.id)
    done = await wait_for_status(pool, claimed.id, {"parsed", "failed"})

    assert claimed.status == "parsing" and done.status == "parsed"
    assert [b.text for b in await repo.list_blocks(pool, done.id)] == ["one paragraph"]


async def test_reparse_of_a_missing_document_or_one_being_parsed_is_refused(
    pool, service, document
):
    with pytest.raises(LookupError):
        await service.reparse(uuid4())
    await repo.claim_for_parsing(pool, document.id)  # a worker has it

    with pytest.raises(ParseBusyError):
        await service.reparse(document.id)


async def test_parses_never_run_more_at_once_than_the_limit(
    pool, settings, monkeypatch
):
    limited = DocumentService(
        pool, settings.model_copy(update={"max_concurrent_parses": 1})
    )
    lock = threading.Lock()
    state = {"now": 0, "max": 0}
    original = documents_service.parse_file

    def tracked(*args, **kwargs):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        try:
            time.sleep(0.15)
            return original(*args, **kwargs)
        finally:
            with lock:
                state["now"] -= 1

    monkeypatch.setattr(documents_service, "parse_file", tracked)
    try:
        results = [
            await upload(
                limited,
                mf.docx_bytes(lambda d, n=n: d.add_paragraph(f"doc {n}")),
                f"{n}.docx",
            )
            for n in range(3)
        ]
        for result in results:
            await wait_for_status(pool, result.document.id, {"parsed"})
    finally:
        await limited.close()

    assert state["max"] == 1
