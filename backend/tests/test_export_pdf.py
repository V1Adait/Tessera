"""The PDF export: what is on the pages, what it refuses, and the route that serves it.

The builder is pure, so it is tested without a database by reading the PDF back with pdfplumber.
Only the route tests need the database (conftest); they use no model and no document parsing.
"""

import io
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pdfplumber
import pytest

from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import EntityRecord
from backend.app.export.pdf import (
    MAX_COLUMNS,
    MAX_ROWS,
    ExportTooLargeError,
    _safe,
    build_pdf,
)
from backend.app.main import app
from backend.app.matrix.models import (
    CellOutcome,
    CellRecord,
    ColumnDraft,
    ColumnRecord,
    EvidenceRecord,
)

NOW = datetime(2026, 10, 17, 12, 30, tzinfo=UTC)
DOC = uuid4()
FILES = {DOC: "q2-update.pdf"}


def entity(name="Palantir", ticker="PLTR") -> EntityRecord:
    return EntityRecord(
        id=uuid4(),
        workspace_id=uuid4(),
        name=name,
        position=1,
        created_at=NOW,
        ticker=ticker,
    )


def column(name="Revenue", type_="currency") -> ColumnRecord:
    return ColumnRecord(
        id=uuid4(),
        workspace_id=uuid4(),
        position=1,
        version=1,
        name=name,
        prompt="q?",
        type=type_,
        allowed_values=[],
        applicability=None,
        evidence_required=True,
        depends_on=[],
        created_at=NOW,
        updated_at=NOW,
    )


def cell(
    row,
    col,
    status="answered",
    value="$2.0 billion",
    quote="Revenue was $2.0 billion.",
    **kw,
) -> CellRecord:
    cell_id = uuid4()
    evidence = (
        [
            EvidenceRecord(
                id=uuid4(),
                cell_id=cell_id,
                seq=0,
                document_id=DOC,
                chunk_id=None,
                page_start=kw.get("page", 22),
                page_end=None,
                char_start=0,
                char_end=5,
                exact_range=True,
                quote=quote,
            )
        ]
        if quote
        else []
    )
    return CellRecord(
        id=cell_id,
        workspace_id=row.workspace_id,
        entity_id=row.id,
        column_id=col.id,
        column_version=1,
        status=status,
        status_detail=kw.get("detail"),
        value=None,
        display_value=value,
        as_of=None,
        derivation=kw.get("derivation"),
        stale=kw.get("stale", False),
        pinned=False,
        model=None,
        prompt_version=None,
        created_at=NOW,
        updated_at=NOW,
        evidence=evidence,
        period=kw.get("period", "Q2 2026"),
        checks=kw.get("checks", []),
    )


def pdf(rows, columns, cells, documents=None, name="PE diligence") -> bytes:
    return build_pdf(name, rows, columns, cells, documents or {}, FILES, NOW)


def pages(data: bytes) -> list[str]:
    with pdfplumber.open(io.BytesIO(data)) as doc:
        return [" ".join((p.extract_text() or "").split()) for p in doc.pages]


def text(data: bytes) -> str:
    return " ".join(pages(data))


# --- a normal export -------------------------------------------------------------------------------------------------


def test_it_is_a_real_pdf_with_the_workspace_title_and_a_summary_then_sources():
    r, c = entity(), column()

    data = pdf([r], [c], [cell(r, c)], {r.id: ["q2-update.pdf"]})

    assert data.startswith(b"%PDF")
    first = pages(data)[0]
    assert "PE diligence" in first and "Palantir" in first and "PLTR" in first
    assert "$2.0 billion" in first and "Q2 2026" in first
    assert "1 companies" in first and "1 questions" in first
    assert "Sources" in text(data)


def test_the_sources_quote_the_words_the_file_and_the_page():
    r, c = entity(), column()

    body = text(
        pdf(
            [r],
            [c],
            [cell(r, c, quote="Q2 2026 revenue was $2.0 billion, up 48%.")],
            {r.id: ["q2-update.pdf"]},
        )
    )

    assert "Q2 2026 revenue was $2.0 billion, up 48%." in body
    assert "q2-update.pdf, p. 22" in body
    assert "Documents: q2-update.pdf" in body


def test_a_computed_cell_says_so_and_shows_its_formula_and_inputs():
    r, margin = entity(), column("Operating margin", "number")
    derivation = {
        "op": "ratio",
        "inputs": [
            {
                "column": "Operating income",
                "display": "$500 million",
                "period": "Q2 2026",
            },
            {"column": "Revenue", "display": "$2,000 million", "period": "Q2 2026"},
        ],
        "notes": [],
    }

    body = text(
        pdf([r], [margin], [cell(r, margin, value="25%", derivation=derivation)])
    )

    assert "computed" in body
    assert "Computed: Operating income / Revenue" in body
    assert "Operating income: $500 million (Q2 2026)" in body


def test_a_failed_tie_out_is_shown_with_its_detail_and_a_pass_is_shown_too():
    r, gross = entity(), column("Gross profit")
    fails = [
        {
            "id": "g",
            "name": "Gross profit ties",
            "status": "does_not_tie",
            "detail": "a gap of $50 million (4.2%)",
        }
    ]
    passes = [
        {"id": "g", "name": "Gross profit ties", "status": "ties", "detail": "agrees"}
    ]

    bad = text(pdf([r], [gross], [cell(r, gross, checks=fails)]))
    good = text(pdf([r], [gross], [cell(r, gross, checks=passes)]))

    assert "does not tie out" in bad and "a gap of $50 million (4.2%)" in bad
    assert "ties out" in good and "does not tie" not in good


def test_an_unverified_answer_is_marked_unverified_and_never_as_a_plain_value():
    r, c = entity(), column()

    body = text(
        pdf(
            [r],
            [c],
            [
                cell(
                    r,
                    c,
                    "low_confidence",
                    value="$9",
                    quote=None,
                    detail="no quote backed it",
                )
            ],
        )
    )

    assert "Unverified: $9" in body and "no quote backed it" in body


def test_cells_that_are_not_answered_say_why_and_are_listed_as_missing():
    r = entity()
    a, b, c = column("Cash"), column("Debt"), column("Covenants")

    body = text(
        pdf(
            [r],
            [a, b, c],
            [
                cell(r, a, "not_found", value=None, quote=None),
                cell(r, b, "pending_or_error", value=None, quote=None),
            ],
        )
    )

    assert "Not found" in body and "Could not be read" in body and "Not filled" in body
    assert (
        "Not answered: Cash (not found); Debt (could not be read); Covenants (not filled)"
        in body
    )


# --- layout limits --------------------------------------------------------------------------------------------------------


def test_many_questions_are_split_into_blocks_so_the_text_stays_readable():
    r = entity()
    columns = [column(f"Metric {i}") for i in range(14)]

    body = text(
        pdf([r], columns, [cell(r, c, value=f"${i}") for i, c in enumerate(columns)])
    )

    assert "Questions 1 to 6 of 14" in body
    assert "Questions 7 to 12 of 14" in body
    assert "Questions 13 to 14 of 14" in body


def test_a_very_long_quote_is_cut_not_allowed_to_fill_the_page():
    r, c = entity(), column()

    body = text(pdf([r], [c], [cell(r, c, quote="word " * 400)]))

    assert body.count("word") < 80 and "..." in body


def test_an_empty_matrix_still_makes_a_pdf_that_says_so():
    data = pdf([], [], [])

    assert data.startswith(b"%PDF") and "nothing to show yet" in text(data)


@pytest.mark.parametrize(("rows", "cols"), [(MAX_ROWS + 1, 1), (1, MAX_COLUMNS + 1)])
def test_a_matrix_too_big_to_read_on_paper_is_refused_with_advice(rows, cols):
    with pytest.raises(ExportTooLargeError, match="Remove some, or export fewer"):
        pdf(
            [entity(f"C{i}") for i in range(rows)],
            [column(f"Q{i}") for i in range(cols)],
            [],
        )


# --- hostile and unusual text ----------------------------------------------------------------------------------------------


def test_text_that_looks_like_markup_is_printed_literally_never_interpreted():
    r = entity("Smith & <b>Sons</b>")
    c = column("Revenue <i>")

    body = text(
        pdf(
            [r],
            [c],
            [
                cell(
                    r, c, value="$1 <script>", quote="</para> & more <font color='red'>"
                )
            ],
        )
    )

    assert "Smith & <b>Sons</b>" in body
    assert "<script>" in body and "</para>" in body


def test_characters_the_pdf_font_cannot_draw_become_question_marks_not_a_crash():
    r, c = entity("Société Générale"), column()

    data = pdf(
        [r], [c], [cell(r, c, value="≥ $5 \U0001f680", quote="growth ≥ 5% → up − 3")]
    )

    body = text(data)
    assert "Soci" in body and ">= $5" in body and "up - 3" in body and "->" in body


def test_safe_text_maps_common_symbols_and_never_raises():
    assert _safe("a−b ≥ c d") == "a-b >= c d"
    assert _safe("\U0001f680") == "?"
    assert _safe("café “quoted”") == "café “quoted”"  # these ARE in the font


# --- the route -----------------------------------------------------------------------------------------------------------------


@pytest.fixture
async def client(pool) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_pool] = lambda: pool
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def seeded(pool) -> tuple[UUID, str]:
    """A workspace with one company, one question and one answered cell (no document parsing)."""
    workspace = await repo.default_workspace_id(pool)
    row = await repo.create_entity(pool, workspace, "Palantir", "PLTR")
    col = await mrepo.create_column(
        pool,
        workspace,
        ColumnDraft(
            name="Revenue",
            prompt="What was revenue?",
            type="currency",
            evidence_required=False,
        ),
    )
    await mrepo.save_cell(
        pool,
        workspace_id=workspace,
        entity_id=row.id,
        column=col,
        outcome=CellOutcome(
            status="answered",
            value={"amount": "2000000000", "currency": "USD"},
            display="$2 billion",
            period="Q2 2026",
        ),
    )
    name = await pool.fetchval("select name from workspaces where id = $1", workspace)
    return workspace, name


async def test_the_route_serves_a_pdf_as_a_download_that_is_not_cached(client, pool):
    workspace, _ = await seeded(pool)

    r = await client.get(f"/workspaces/{workspace}/export/pdf")

    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["content-disposition"].startswith('attachment; filename="tessera-')
    assert r.headers["content-disposition"].endswith('.pdf"')
    body = text(r.content)
    assert "Palantir" in body and "$2 billion" in body and "Q2 2026" in body


async def test_the_file_name_is_plain_ascii_whatever_the_workspace_is_called(
    client, pool
):
    workspace, _ = await seeded(pool)
    await pool.execute(
        "update workspaces set name = $2 where id = $1",
        workspace,
        'PE "Q3" / Fund <A>é’s',
    )

    r = await client.get(f"/workspaces/{workspace}/export/pdf")

    header = r.headers["content-disposition"]
    assert (
        header.isascii()
        and "/" not in header.split("filename=")[1]
        and "<" not in header
    )
    assert "tessera-pe-q3-fund-a-s-" in header


async def test_the_route_404s_for_an_unknown_workspace(client):
    assert (await client.get(f"/workspaces/{uuid4()}/export/pdf")).status_code == 404


async def test_the_route_422s_with_advice_when_the_matrix_is_too_big(
    client, pool, monkeypatch
):
    workspace, _ = await seeded(pool)
    monkeypatch.setattr("backend.app.export.pdf.MAX_ROWS", 0)

    r = await client.get(f"/workspaces/{workspace}/export/pdf")

    assert r.status_code == 422 and "Remove some, or export fewer" in r.json()["detail"]
