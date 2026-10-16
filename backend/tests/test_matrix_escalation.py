"""Looking again with a wider net, and what a fill costs.

Found on the Apple 10-K: the answer to two columns ranked 10th and 11th, one place outside the top 5
parents the model is shown. A "not found", or an answer the code does not trust, earns ONE wider
look. These tests use a scripted model so every branch is exact. Needs the database (conftest).
"""

from decimal import Decimal

import pytest

from backend.app.documents_service import DocumentService
from backend.app.matrix.extractor import LabelledChunk, decide, outcome_rank
from backend.app.matrix.models import CellOutcome, ModelReply, ProposedEvidence
from backend.tests import make_files as mf
from backend.tests.fakes import make_settings
from backend.tests.test_matrix_decide import column, labelled, table_chunk
from backend.tests.test_matrix_flow import (
    EMPLOYEES,
    World,
    good,
    parsed,
    reply,
)

STRONG = "The company has many employees: employees work for the company and people work every day."
ANSWER = (
    "As of year end the firm had approximately 166,000 full-time equivalent employees."
)
ANSWER_QUOTE = "approximately 166,000 full-time equivalent employees"
OTHER = "Other employees were hired to work in the canteen."  # matches the search, answers nothing
FLAGGED_FIRST = "The company has approximately 170,000 employees; employees work for the company, people work."
FLAG_SQL = (
    "update document_chunks set confidence = 0.5, flags = ARRAY['table_total_mismatch:r1c1'] "
    "where text like '%170,000%'"
)


@pytest.fixture
def settings(tmp_path):
    # no parent merging: these tests need the two small sections to stay separate parents
    return make_settings().model_copy(
        update={"upload_dir": str(tmp_path / "uploads"), "chunk_parent_min_tokens": 0}
    )


@pytest.fixture
async def documents(pool, settings):
    svc = DocumentService(pool, settings)
    yield svc
    await svc.close()


@pytest.fixture
async def world(pool, settings, documents) -> World:
    return await World(pool, settings, documents).start()


def sections_docx(sections: list[tuple[str, str]]) -> bytes:
    def build(d):
        for heading, text in sections:
            d.add_heading(heading, 1)
            d.add_paragraph(text)

    return mf.docx_bytes(build)


def narrow(world: World, **overrides):
    """Show the model only the single best parent at first; a wider look sees ten."""
    world.service._extractor._settings = world.settings.model_copy(  # type: ignore[union-attr]
        update={
            "retrieval_max_parents": 1,
            "extract_escalation_parents": 10,
            **overrides,
        }
    )


async def two_sections(world: World, first=STRONG, second=ANSWER):
    return await parsed(
        world.documents,
        world.pool,
        sections_docx([("Overview", first), ("Notes", second)]),
        entity_id=world.acme.id,
    )


def not_found(reason="The shown passages do not say.") -> str:
    return reply(status="not_found", reason=reason)


# --- the wider look helps ---------------------------------------------------------------------


async def test_a_not_found_gets_a_wider_look_that_finds_the_answer(world):
    await two_sections(world)
    narrow(world)
    column_ = await world.column()
    world.script(not_found(), good(quote=ANSWER_QUOTE, chunk="C2"))

    cell = await world.fill(column_)

    assert cell.status == "answered" and cell.value == {
        "number": "166000",
        "unit": None,
    }
    assert len(world.llm.seen) == 2
    first_prompt, second_prompt = (m[1].content for m in world.llm.seen)
    assert "166,000" not in first_prompt and "166,000" in second_prompt


async def test_an_unreliable_source_gets_a_wider_look_that_finds_a_better_one(world):
    await two_sections(world, first=FLAGGED_FIRST)
    await world.pool.execute(FLAG_SQL)
    narrow(world)
    column_ = await world.column()
    world.script(
        good(answer="170,000", quote="approximately 170,000 employees"),
        good(quote=ANSWER_QUOTE, chunk="C2"),
    )

    cell = await world.fill(column_)

    assert cell.status == "answered" and cell.value == {
        "number": "166000",
        "unit": None,
    }
    assert [e.quote for e in cell.evidence] == [ANSWER_QUOTE]


# --- the wider look is not wasted, and cannot hurt --------------------------------------------


async def test_the_wider_look_costs_nothing_when_it_would_show_nothing_new(world):
    await world.acme_document(
        "There are many employees and people who work at the company."
    )
    column_ = await world.column()
    world.script(not_found())

    cell = await world.fill(column_)

    assert cell.status == "not_found"
    assert len(world.llm.seen) == 1  # the one parent was already everything there is


async def test_a_second_look_that_finds_nothing_leaves_the_first_outcome(world):
    await two_sections(world, second=OTHER)
    narrow(world)
    column_ = await world.column()
    world.script(
        not_found("First look found nothing."),
        not_found("Second look found nothing either."),
    )

    cell = await world.fill(column_)

    assert (
        cell.status == "not_found" and cell.status_detail == "First look found nothing."
    )
    assert len(world.llm.seen) == 2


async def test_a_failing_second_look_cannot_make_the_cell_worse(world):
    await two_sections(world)
    narrow(world)
    column_ = await world.column()
    world.script(not_found(), RuntimeError("provider down"))

    cell = await world.fill(column_)

    assert cell.status == "not_found"  # not pending_or_error: the first answer stands


async def test_if_the_wider_look_finds_nothing_better_the_unreliable_answer_stands(
    world,
):
    await two_sections(world, first=FLAGGED_FIRST, second=OTHER)
    await world.pool.execute(FLAG_SQL)
    narrow(world)
    column_ = await world.column()
    bad = good(answer="170,000", quote="approximately 170,000 employees")
    world.script(bad, bad)

    cell = await world.fill(column_)

    assert cell.status == "low_confidence" and cell.value == {
        "number": "170000",
        "unit": None,
    }
    assert "unreliable" in (cell.status_detail or "")
    assert (
        len(world.llm.seen) == 2
    )  # it did look again; the second look found nothing better


async def test_a_wider_search_that_adds_nothing_unseen_makes_no_second_call(world):
    await two_sections(
        world, second="Unrelated text about parking and cars."
    )  # never retrieved
    narrow(world)
    column_ = await world.column()
    world.script(not_found())

    cell = await world.fill(column_)

    assert cell.status == "not_found" and len(world.llm.seen) == 1


async def test_the_wider_look_can_be_switched_off(world):
    await two_sections(world)
    narrow(world, extract_escalate=False)
    column_ = await world.column()
    world.script(not_found(), good(quote=ANSWER_QUOTE, chunk="C2"))

    cell = await world.fill(column_)

    assert cell.status == "not_found" and len(world.llm.seen) == 1


async def test_a_row_with_no_documents_is_not_searched_twice(world):
    column_ = await world.column()
    world.script(not_found())

    cell = await world.fill(column_)

    assert cell.status == "not_found" and world.llm.seen == []


async def test_an_answered_cell_is_never_looked_at_twice(world):
    await world.acme_document()
    column_ = await world.column()
    world.script(good())

    await world.fill(column_)

    assert len(world.llm.seen) == 1


# --- what a fill cost -------------------------------------------------------------------------


async def test_a_fill_knows_what_all_its_model_calls_cost(world):
    await two_sections(world)
    narrow(world)
    column_ = await world.column()
    world.llm.reported_cost = 0.0002
    world.script(not_found(), good(quote=ANSWER_QUOTE, chunk="C2"))

    outcome = await world.service._extractor.extract(  # type: ignore[union-attr]
        world.workspace, world.acme.id, "Acme", column_
    )

    assert outcome.status == "answered"
    assert outcome.model_calls == 2 and outcome.cost_usd == Decimal("0.0004")
    assert outcome.unknown_cost_calls == 0


async def test_calls_with_no_known_cost_are_counted_not_treated_as_free(world):
    await world.acme_document()
    column_ = await world.column()
    world.script(good())

    outcome = await world.service._extractor.extract(  # type: ignore[union-attr]
        world.workspace, world.acme.id, "Acme", column_
    )

    assert outcome.cost_usd == Decimal(0) and outcome.unknown_cost_calls == 1


async def test_a_row_with_no_documents_costs_nothing(world):
    column_ = await world.column()

    outcome = await world.service._extractor.extract(  # type: ignore[union-attr]
        world.workspace, world.acme.id, "Acme", column_
    )

    assert outcome.model_calls == 0 and outcome.cost_usd == Decimal(0)


# --- pure rules -------------------------------------------------------------------------------


def _decide(rep: ModelReply, col, context):
    return decide(rep, col, context, low_confidence_at=0.7, final_attempt=False)


def test_untrustworthy_answers_ask_for_a_wider_look_but_good_ones_do_not():
    quote = "approximately 166,000 full-time equivalent employees"
    rep = ModelReply(
        status="answered",
        answer="166,000",
        evidence=[ProposedEvidence(chunk="C1", quote=quote)],
    )
    good_answer = _decide(rep, column("number"), labelled(EMPLOYEES))
    weak = _decide(rep, column("number"), labelled(EMPLOYEES, confidence=0.5))
    unscaled = _decide(
        ModelReply(
            status="answered",
            answer="$78,328",
            evidence=[ProposedEvidence(chunk="C1", quote="term debt |  | $78,328")],
        ),
        column("currency", name="Term debt", prompt="Term debt"),
        [
            LabelledChunk(
                "C1",
                table_chunk("Total non-current portion of term debt |  | $78,328"),
                "f.html",
            )
        ],
    )

    assert good_answer.outcome and good_answer.outcome.retry_wider is False
    assert (
        weak.outcome
        and weak.outcome.status == "low_confidence"
        and weak.outcome.retry_wider
    )
    assert unscaled.outcome and unscaled.outcome.retry_wider


def test_a_better_outcome_ranks_higher():
    answered = CellOutcome(status="answered", value=1, display="1")
    with_value = CellOutcome(status="low_confidence", value=1, display="1")
    bare = CellOutcome(status="low_confidence")
    nothing = CellOutcome(status="not_found")
    broken = CellOutcome(status="pending_or_error")

    assert outcome_rank(answered) > outcome_rank(with_value) > outcome_rank(bare)
    assert outcome_rank(bare) >= outcome_rank(nothing) > outcome_rank(broken)
