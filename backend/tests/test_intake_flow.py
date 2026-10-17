"""Intake end to end: a real file is uploaded and parsed, a SCRIPTED model reads its first page, and
the row appears (or a question is asked). Real parser, real database. Skips without a database.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest

from backend.app.api_documents import get_documents
from backend.app.config import get_settings
from backend.app.db import intake_repository as irepo
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.documents_service import DocumentService
from backend.app.intake.identify import IdentityExtractor
from backend.app.intake.service import IntakeService
from backend.app.main import app
from backend.app.tracing import db_recorder
from backend.tests import make_files as mf
from backend.tests.fakes import QueueFake, make_settings
from backend.tests.test_chunks_db import reader, wait_final

PALANTIR = (
    "Palantir Technologies Inc. reported results. Class A common stock PLTR Nasdaq."
)
ACME = "Acme Corporation announced quarterly results for the period."


def docx(*paragraphs: str) -> bytes:
    def build(d):
        d.add_heading("Report", 1)
        for text in paragraphs:
            d.add_paragraph(text)

    return mf.docx_bytes(build)


def says(
    company="Palantir Technologies Inc.", ticker="PLTR", confidence=0.95, **kw
) -> str:
    return json.dumps(
        {
            "company": company,
            "ticker": ticker,
            "doc_type": "quarterly report",
            "period": "Q2 2026",
            "confidence": confidence,
            **kw,
        }
    )


@pytest.fixture
def settings(tmp_path):
    return make_settings().model_copy(update={"upload_dir": str(tmp_path / "uploads")})


class Setup:
    def __init__(self, pool, settings):
        self.pool, self.settings = pool, settings
        self.llm = QueueFake(replies=[says()])
        self.documents = DocumentService(pool, settings)
        self.intake = IntakeService(
            pool,
            settings,
            IdentityExtractor(
                self.llm,
                provider="fake",
                model="scripted",
                recorder=db_recorder(pool),
            ),
            self.documents.assign_entity,
        )
        self.documents.on_parsed = self.intake.process

    async def start(self):
        self.workspace = await repo.default_workspace_id(self.pool)
        return self

    async def upload(self, text: str, name="report.docx", entity_id=None):
        result = await self.documents.upload(
            workspace_id=None,
            entity_id=entity_id,
            filename=name,
            read=reader(docx(text)),
        )
        return await wait_final(self.pool, result.document.id)

    async def intake_of(self, document_id, wait=5.0):
        """The recorded decision, waiting for the background step to finish."""
        deadline = asyncio.get_running_loop().time() + wait
        while asyncio.get_running_loop().time() < deadline:
            found = await irepo.get_result(self.pool, document_id)
            if found is not None:
                return found
            await asyncio.sleep(0.03)
        raise AssertionError("intake never recorded a decision")

    async def rows(self):
        return await repo.list_entities(self.pool, self.workspace)


@pytest.fixture
async def setup(pool, settings) -> AsyncIterator[Setup]:
    s = await Setup(pool, settings).start()
    yield s
    await s.documents.close()


# --- the happy paths ----------------------------------------------------------------------------------


async def test_a_file_about_a_new_company_creates_its_row_and_attaches_the_file(setup):
    doc = await setup.upload(PALANTIR)

    result = await setup.intake_of(doc.id)

    assert result.outcome == "created"
    rows = await setup.rows()
    assert [(r.name, r.ticker) for r in rows] == [("Palantir Technologies", "PLTR")]
    assert (await repo.get_document(setup.pool, doc.id)).entity_id == rows[0].id  # type: ignore[union-attr]
    assert result.detected["company"] == "Palantir Technologies Inc."


async def test_a_second_file_about_the_same_company_joins_the_same_row(setup):
    first = await setup.upload(PALANTIR, "q2.docx")
    await setup.intake_of(first.id)
    setup.llm.replies[:] = [says(company="Palantir", ticker="PLTR")]

    second = await setup.upload(PALANTIR + " Annual figures.", "10k.docx")
    result = await setup.intake_of(second.id)

    assert result.outcome == "assigned"
    assert len(await setup.rows()) == 1
    docs = await repo.list_documents(
        setup.pool, setup.workspace, entity_id=(await setup.rows())[0].id
    )
    assert {d.filename for d in docs} == {"q2.docx", "10k.docx"}


async def test_the_next_spelling_is_remembered_as_an_alias(setup):
    await setup.intake_of((await setup.upload(PALANTIR)).id)
    setup.llm.replies[:] = [
        says(company="Palantir Technologies Incorporated", ticker=None)
    ]
    other = await setup.upload(PALANTIR + " More.", "b.docx")

    await setup.intake_of(other.id)

    row = (await setup.rows())[0]
    assert (
        "Palantir Technologies Incorporated" not in row.aliases
    )  # same name once suffix-stripped
    setup.llm.replies[:] = [says(company="PLTR Corp", ticker="PLTR")]
    third = await setup.upload(PALANTIR + " Even more.", "c.docx")
    await setup.intake_of(third.id)
    assert (
        "PLTR Corp" in (await setup.rows())[0].aliases
    )  # matched by ticker; spelling learned


async def test_two_different_companies_make_two_rows(setup):
    await setup.intake_of((await setup.upload(PALANTIR)).id)
    setup.llm.replies[:] = [says(company="Acme Corporation", ticker=None)]

    await setup.intake_of((await setup.upload(ACME, "acme.docx")).id)

    assert [r.name for r in await setup.rows()] == ["Palantir Technologies", "Acme"]


# --- a person's choice and unsure cases ---------------------------------------------------------------------


async def test_a_file_attached_to_a_row_by_a_person_is_left_alone(setup):
    mine = await repo.create_entity(setup.pool, setup.workspace, "My Row")

    doc = await setup.upload(PALANTIR, entity_id=mine.id)
    result = await setup.intake_of(doc.id)

    assert result.outcome == "skipped"
    assert len(setup.llm.seen) == 0  # no model call was paid for
    assert (await repo.get_document(setup.pool, doc.id)).entity_id == mine.id  # type: ignore[union-attr]


async def test_an_unsure_model_leaves_the_file_waiting_with_a_reason(setup):
    setup.llm.replies[:] = [says(confidence=0.4)]

    doc = await setup.upload(PALANTIR)
    result = await setup.intake_of(doc.id)

    assert result.outcome == "needs_review" and "not sure" in (result.detail or "")
    assert await setup.rows() == []
    assert (await repo.get_document(setup.pool, doc.id)).entity_id is None  # type: ignore[union-attr]


async def test_a_company_the_text_does_not_mention_is_never_acted_on(setup):
    setup.llm.replies[:] = [says(company="Microsoft Corporation", ticker="MSFT")]

    result = await setup.intake_of((await setup.upload(PALANTIR)).id)

    assert result.outcome == "needs_review" and await setup.rows() == []


async def test_a_look_alike_row_is_a_question_not_a_merge(setup):
    await repo.create_entity(setup.pool, setup.workspace, "Apple")
    setup.llm.replies[:] = [says(company="Apple Hospitality REIT", ticker="APLE")]

    result = await setup.intake_of(
        (await setup.upload("Apple Hospitality REIT announced results. APLE NYSE.")).id
    )

    assert result.outcome == "needs_review" and result.candidates == ["Apple"]
    assert [r.name for r in await setup.rows()] == ["Apple"]


async def test_a_failing_model_leaves_the_file_waiting(setup):
    setup.llm.replies[:] = [RuntimeError("provider down")]

    result = await setup.intake_of((await setup.upload(PALANTIR)).id)

    assert result.outcome == "needs_review" and "could not tell" in (
        result.detail or ""
    )


async def test_no_model_at_all_leaves_files_waiting_and_says_why(pool, settings):
    s = await Setup(pool, settings).start()
    s.intake._identifier = None
    try:
        result = await s.intake_of((await s.upload(PALANTIR)).id)
    finally:
        await s.documents.close()

    assert result.outcome == "needs_review" and "no model" in (result.detail or "")


# --- races and repeats --------------------------------------------------------------------------------------


async def test_two_files_about_a_new_company_at_once_make_one_row(setup):
    setup.llm.replies[:] = [says()]  # the last reply repeats for every call
    a = await setup.documents.upload(
        workspace_id=None,
        entity_id=None,
        filename="a.docx",
        read=reader(docx(PALANTIR + " A.")),
    )
    b = await setup.documents.upload(
        workspace_id=None,
        entity_id=None,
        filename="b.docx",
        read=reader(docx(PALANTIR + " B.")),
    )

    await asyncio.gather(setup.intake_of(a.document.id), setup.intake_of(b.document.id))

    assert len(await setup.rows()) == 1
    docs = await repo.list_documents(
        setup.pool, setup.workspace, entity_id=(await setup.rows())[0].id
    )
    assert len(docs) == 2


async def test_a_reparse_does_not_rewrite_the_decision(setup):
    doc = await setup.upload(PALANTIR)
    first = await setup.intake_of(doc.id)

    again = await setup.intake.process(doc.id)

    assert again is not None and again.outcome == first.outcome == "created"
    assert len(setup.llm.seen) == 1


async def test_intake_is_not_run_for_a_file_that_has_not_been_parsed(setup, pool):
    result = await setup.intake.process(uuid4())

    assert result is None


async def test_the_intake_call_is_traced_with_its_kind(setup, pool):
    await setup.intake_of((await setup.upload(PALANTIR)).id)

    row = await pool.fetchrow(
        "select kind, status, prompt_version from traces where kind = 'intake'"
    )
    assert (
        row is not None
        and row["status"] == "ok"
        and row["prompt_version"] == "intake-2"
    )


async def test_an_error_in_intake_never_turns_a_good_parse_into_a_failed_one(setup):
    async def boom(_id):
        raise RuntimeError("intake exploded")

    setup.documents.on_parsed = boom

    doc = await setup.upload(PALANTIR)

    assert doc.status == "parsed"


# --- the API -----------------------------------------------------------------------------------------------------


@pytest.fixture
async def client(pool, setup) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_settings] = lambda: setup.settings
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_documents] = lambda: setup.documents
    app.state.intake = setup.intake
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.intake = None


async def test_the_api_says_where_a_file_went(setup, client):
    doc = await setup.upload(PALANTIR)
    await setup.intake_of(doc.id)

    r = await client.get(f"/documents/{doc.id}/intake")

    assert r.status_code == 200
    body = r.json()
    assert (
        body["outcome"] == "created" and body["entity_name"] == "Palantir Technologies"
    )
    assert body["detected"]["ticker"] == "PLTR"


async def test_the_api_follows_a_manual_change_of_row(setup, client):
    setup.llm.replies[:] = [says(confidence=0.3)]
    doc = await setup.upload(PALANTIR)
    await setup.intake_of(doc.id)
    mine = await repo.create_entity(setup.pool, setup.workspace, "Pick Me")
    assert (
        await client.patch(f"/documents/{doc.id}", json={"entity_id": str(mine.id)})
    ).status_code == 200

    body = (await client.get(f"/documents/{doc.id}/intake")).json()

    assert body["outcome"] == "needs_review"  # what intake decided
    assert body["entity_name"] == "Pick Me"  # where the file is now


async def test_the_api_404s_before_intake_has_looked(setup, client):
    r = await client.get(f"/documents/{uuid4()}/intake")

    assert r.status_code == 404


async def test_the_api_can_run_intake_again(setup, client):
    setup.llm.replies[:] = [says(confidence=0.3)]
    doc = await setup.upload(PALANTIR)
    await setup.intake_of(doc.id)
    setup.llm.replies[:] = [says()]

    r = await client.post(f"/documents/{doc.id}/intake")

    assert r.status_code == 200 and r.json()["outcome"] == "created"


# --- the website the first page prints becomes the company's domain --------------------------------------------------------


async def test_a_new_row_gets_the_website_the_page_prints(setup):
    page = PALANTIR + " Investor relations: www.palantir.com"
    setup.llm.replies[:] = [says(domain="palantir.com")]

    await setup.intake_of((await setup.upload(page)).id)

    rows = await setup.rows()
    assert [(r.name, r.domain) for r in rows] == [
        ("Palantir Technologies", "palantir.com")
    ]


async def test_a_website_the_page_does_not_print_is_not_stored(setup):
    setup.llm.replies[:] = [
        says(domain="palantir.com")
    ]  # the model remembers it; the page has none

    await setup.intake_of((await setup.upload(PALANTIR)).id)

    assert (await setup.rows())[0].domain is None


async def test_a_later_file_fills_in_a_missing_website_but_never_overwrites_one(setup):
    await setup.intake_of((await setup.upload(PALANTIR)).id)
    setup.llm.replies[:] = [says(domain="palantir.com")]
    await setup.intake_of(
        (await setup.upload(PALANTIR + " IR: www.palantir.com", "b.docx")).id
    )
    first = (await setup.rows())[0].domain
    setup.llm.replies[:] = [says(domain="other.com")]

    await setup.intake_of(
        (await setup.upload(PALANTIR + " also www.other.com", "c.docx")).id
    )

    assert first == "palantir.com" and (await setup.rows())[0].domain == "palantir.com"
