"""The /documents routes over HTTP, against a real database and the real parsers."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from backend.app.api_documents import get_documents
from backend.app.config import get_settings
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.documents_service import DocumentService
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


@pytest.fixture
async def client(pool, settings, service) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_documents] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


def files(data: bytes, name: str = "report.docx"):
    return {"file": (name, data, "application/octet-stream")}


async def wait_final(client, document_id, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        body = (await client.get(f"/documents/{document_id}")).json()
        if body["status"] in FINAL:
            return body
        await asyncio.sleep(0.05)
    raise AssertionError("the document never finished parsing")


def a_docx(text="Revenue grew strongly."):
    def build(d):
        d.add_heading("Annual report", 1)
        d.add_paragraph(text)

    return mf.docx_bytes(build)


# --- the happy path ---------------------------------------------------------------------


async def test_upload_returns_at_once_then_the_document_is_parsed(client):
    response = await client.post("/documents", files=files(a_docx()))

    assert response.status_code == 201
    body = response.json()
    assert (
        body["status"] == "uploaded"
        and body["kind"] == "docx"
        and body["duplicate"] is False
    )
    assert body["filename"] == "report.docx" and len(body["sha256"]) == 64

    done = await wait_final(client, body["id"])
    assert (
        done["status"] == "parsed"
        and done["block_count"] == 2
        and done["parser"] == "docx/python-docx"
    )
    assert done["report"]["block_counts"] == {"heading": 1, "paragraph": 1}


async def test_the_parsed_blocks_are_readable_in_order_and_can_be_filtered_and_paged(
    client,
):
    document = (await client.post("/documents", files=files(a_docx()))).json()
    await wait_final(client, document["id"])

    blocks = (await client.get(f"/documents/{document['id']}/blocks")).json()
    assert [(b["seq"], b["type"], b["text"]) for b in blocks] == [
        (0, "heading", "Annual report"),
        (1, "paragraph", "Revenue grew strongly."),
    ]
    assert (
        blocks[1]["location"]["section_path"] == ["Annual report"]
        and blocks[1]["char_start"] > blocks[0]["char_end"]
    )

    headings = (
        await client.get(
            f"/documents/{document['id']}/blocks", params={"type": "heading"}
        )
    ).json()
    assert [b["type"] for b in headings] == ["heading"]
    page = (
        await client.get(
            f"/documents/{document['id']}/blocks", params={"limit": 1, "offset": 1}
        )
    ).json()
    assert [b["seq"] for b in page] == [1]


async def test_a_pdf_with_a_table_comes_back_with_structured_cells(client):
    rows = [["Region", "Customers"], ["EU", "5,000"], ["US", "9,000"]]
    data = mf.pdf_flowables(
        [mf.para("(in thousands)"), mf.spacer(), mf.grid_table(rows)]
    )
    document = (await client.post("/documents", files=files(data, "table.pdf"))).json()

    done = await wait_final(client, document["id"])

    assert done["status"] == "parsed" and done["page_count"] == 1
    (table,) = (
        await client.get(
            f"/documents/{document['id']}/blocks", params={"type": "table"}
        )
    ).json()
    assert (
        table["data"]["rows"] == rows
        and table["data"]["units"] == "in thousands"
        and table["location"]["page"] == 1
    )


async def test_the_same_file_twice_is_a_duplicate_with_status_200(client):
    first = await client.post("/documents", files=files(a_docx()))
    await wait_final(client, first.json()["id"])

    second = await client.post("/documents", files=files(a_docx(), "other name.docx"))

    assert second.status_code == 200
    assert (
        second.json()["duplicate"] is True and second.json()["id"] == first.json()["id"]
    )
    assert second.json()["filename"] == "report.docx"


async def test_documents_default_to_the_default_workspace_and_can_be_listed(
    client, pool
):
    workspace = await repo.default_workspace_id(pool)
    first = (
        await client.post("/documents", files=files(a_docx("one"), "one.docx"))
    ).json()
    second = (
        await client.post("/documents", files=files(a_docx("two"), "two.docx"))
    ).json()

    listing = (await client.get("/documents")).json()

    assert {d["id"] for d in listing} == {first["id"], second["id"]}
    assert all(d["workspace_id"] == str(workspace) for d in listing)
    assert len((await client.get("/documents", params={"limit": 1})).json()) == 1
    for document in (first, second):
        await wait_final(client, document["id"])


async def test_a_document_can_be_uploaded_to_an_entity_and_listed_by_entity(
    client, pool
):
    workspace = await repo.default_workspace_id(pool)
    entity = await repo.create_entity(pool, workspace, "Alphabet")

    document = (
        await client.post(
            "/documents", data={"entity_id": str(entity.id)}, files=files(a_docx())
        )
    ).json()
    await wait_final(client, document["id"])

    assert document["entity_id"] == str(entity.id)
    assert [
        d["id"]
        for d in (
            await client.get("/documents", params={"entity_id": str(entity.id)})
        ).json()
    ] == [document["id"]]
    assert (
        await client.get("/documents", params={"entity_id": str(uuid4())})
    ).json() == []


# --- refusals ---------------------------------------------------------------------------


async def test_a_wrong_extension_is_415(client):
    response = await client.post(
        "/documents", files=files(mf.pdf_text("x"), "report.docx")
    )

    assert response.status_code == 415 and "PDF" in response.json()["detail"]


async def test_a_macro_document_is_415(client):
    macro = mf.add_to_zip(mf.docx_bytes(), {"word/vbaProject.bin": b"macro"})

    response = await client.post("/documents", files=files(macro))

    assert response.status_code == 415 and "macro" in response.json()["detail"].lower()


async def test_a_zip_bomb_is_422(client):
    bomb = mf.add_to_zip(
        mf.docx_bytes(), {"word/media/huge.bin": b"\x00" * (30 * 1024 * 1024)}
    )

    assert (await client.post("/documents", files=files(bomb))).status_code == 422


async def test_an_empty_file_is_400(client):
    assert (await client.post("/documents", files=files(b""))).status_code == 400


async def test_an_oversized_upload_is_413(client, settings):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"max_upload_bytes": 1000}
    )

    response = await client.post("/documents", files=files(b"x" * 5_000_000, "big.txt"))

    assert response.status_code == 413


async def test_a_file_just_over_the_limit_is_refused_while_streaming(pool, settings):
    """The declared-size pre-check has slack for the multipart envelope; the stream check is exact."""
    tight = settings.model_copy(update={"max_upload_bytes": 2000})
    service = DocumentService(pool, tight)
    app.dependency_overrides[get_settings] = lambda: tight
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_documents] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            response = await c.post("/documents", files=files(b"a" * 2500, "notes.txt"))
    finally:
        app.dependency_overrides.clear()
        await service.close()

    assert response.status_code == 413


@pytest.mark.parametrize("field", ["workspace_id", "entity_id"])
async def test_a_bad_uuid_in_the_form_is_422(client, field):
    assert (
        await client.post(
            "/documents", data={field: "not-a-uuid"}, files=files(a_docx())
        )
    ).status_code == 422


async def test_a_missing_file_part_is_422(client):
    assert (
        await client.post("/documents", data={"workspace_id": str(uuid4())})
    ).status_code == 422


async def test_unknown_workspace_and_foreign_entity_are_404_and_store_nothing(
    client, pool, settings
):
    other = await repo.create_workspace(pool, "other")
    foreign = await repo.create_entity(pool, other.id, "Visa")

    assert (
        await client.post(
            "/documents", data={"workspace_id": str(uuid4())}, files=files(a_docx())
        )
    ).status_code == 404
    unknown_entity = await client.post(
        "/documents", data={"entity_id": str(uuid4())}, files=files(a_docx())
    )
    foreign_entity = await client.post(
        "/documents", data={"entity_id": str(foreign.id)}, files=files(a_docx())
    )

    assert unknown_entity.status_code == foreign_entity.status_code == 404
    root = Path(settings.upload_dir)
    assert not root.exists() or not [p for p in root.rglob("*") if p.is_file()]


async def test_a_path_traversal_file_name_is_only_a_label(client, settings):
    document = (
        await client.post("/documents", files=files(a_docx(), "../../../etc/evil.docx"))
    ).json()
    await wait_final(client, document["id"])

    assert document["filename"] == "evil.docx"
    stored = [p for p in Path(settings.upload_dir).rglob("*") if p.is_file()]
    assert stored and all("evil" not in str(p) for p in stored)
    assert all(
        Path(settings.upload_dir).resolve() in p.resolve().parents for p in stored
    )


# --- reading things that are not there --------------------------------------------------


async def test_unknown_documents_are_404_and_bad_ids_are_422(client):
    missing = uuid4()
    assert (await client.get(f"/documents/{missing}")).status_code == 404
    assert (await client.get(f"/documents/{missing}/blocks")).status_code == 404
    assert (await client.post(f"/documents/{missing}/reparse")).status_code == 404
    assert (
        await client.patch(f"/documents/{missing}", json={"entity_id": None})
    ).status_code == 404
    assert (await client.get("/documents/not-a-uuid")).status_code == 422


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 501}, {"offset": -1}, {"type": "banner"}]
)
async def test_bad_block_query_parameters_are_422(client, params):
    document = (await client.post("/documents", files=files(a_docx()))).json()
    await wait_final(client, document["id"])

    assert (
        await client.get(f"/documents/{document['id']}/blocks", params=params)
    ).status_code == 422


async def test_listing_documents_of_an_unknown_workspace_is_404(client):
    assert (
        await client.get("/documents", params={"workspace_id": str(uuid4())})
    ).status_code == 404


# --- re-parse and assigning -------------------------------------------------------------


async def test_reparse_is_accepted_and_the_document_ends_parsed_again(client):
    document = (await client.post("/documents", files=files(a_docx()))).json()
    await wait_final(client, document["id"])

    response = await client.post(f"/documents/{document['id']}/reparse")

    assert response.status_code == 202 and response.json()["status"] == "parsing"
    done = await wait_final(client, document["id"])
    assert done["status"] == "parsed" and done["block_count"] == 2


async def test_reparse_while_parsing_is_409(client, pool):
    document = (await client.post("/documents", files=files(a_docx()))).json()
    await wait_final(client, document["id"])
    await repo.set_document_status(pool, document["id"], "parsing")

    response = await client.post(f"/documents/{document['id']}/reparse")

    assert response.status_code == 409


async def test_assigning_and_unassigning_an_entity(client, pool):
    workspace = await repo.default_workspace_id(pool)
    entity = await repo.create_entity(pool, workspace, "Alphabet")
    document = (await client.post("/documents", files=files(a_docx()))).json()
    await wait_final(client, document["id"])

    assigned = await client.patch(
        f"/documents/{document['id']}", json={"entity_id": str(entity.id)}
    )
    assert assigned.status_code == 200 and assigned.json()["entity_id"] == str(
        entity.id
    )
    assert assigned.json()["block_count"] == 2
    unassigned = await client.patch(
        f"/documents/{document['id']}", json={"entity_id": None}
    )
    assert unassigned.json()["entity_id"] is None


async def test_assigning_an_entity_from_another_workspace_is_404(client, pool):
    other = await repo.create_workspace(pool, "other")
    foreign = await repo.create_entity(pool, other.id, "Visa")
    document = (await client.post("/documents", files=files(a_docx()))).json()
    await wait_final(client, document["id"])

    assert (
        await client.patch(
            f"/documents/{document['id']}", json={"entity_id": str(foreign.id)}
        )
    ).status_code == 404
    assert (
        await client.patch(f"/documents/{document['id']}", json={})
    ).status_code == 422


# --- not configured ---------------------------------------------------------------------


async def test_without_document_storage_configured_uploads_are_503(pool, settings):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pool] = lambda: (
        pool
    )  # no get_documents override and no lifespan
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            response = await c.post("/documents", files=files(a_docx()))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
