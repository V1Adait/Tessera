"""Computed cells and tie-outs through the real service and database, with a SCRIPTED model.

The model only fills ordinary cells (revenue, operating income, ...). Computed cells (a margin) and
tie-out results come from code, so they are tested with exact numbers. Needs the database
(conftest); each test takes about a second.
"""

import json
from decimal import Decimal

import pytest

from backend.app.db import matrix_repository as mrepo
from backend.app.documents_service import DocumentService
from backend.app.finance.checks import CheckSpec
from backend.app.finance.derive import Derivation
from backend.app.matrix.models import ColumnDraft, ColumnPatch
from backend.app.matrix.runs import RunManager, RunScope
from backend.tests.fakes import make_settings
from backend.tests.test_matrix_flow import World
from backend.tests.test_matrix_runs import wait_run

TEXT = [
    "Revenue for the quarter ended June 30, 2026 was $2,000 million.",
    "Operating income for the quarter ended June 30, 2026 was $500 million.",
    "Cost of revenue for the quarter ended June 30, 2026 was $800 million.",
    "Gross profit for the quarter ended June 30, 2026 was $1,200 million.",
]
REV, OPINC, COST, GROSS = TEXT


def says(answer: str, quote: str, period: str | None = "Q2 2026") -> str:
    return json.dumps(
        {
            "status": "answered",
            "answer": answer,
            "period": period,
            "evidence": [{"chunk": "C1", "quote": quote}],
        }
    )


@pytest.fixture
def settings(tmp_path):
    return make_settings().model_copy(update={"upload_dir": str(tmp_path / "uploads")})


@pytest.fixture
async def documents(pool, settings):
    svc = DocumentService(pool, settings)
    yield svc
    await svc.close()


@pytest.fixture
async def world(pool, settings, documents) -> World:
    w = await World(pool, settings, documents).start()
    await w.acme_document(*TEXT)
    return w


async def money(world: World, name: str):
    return await world.column(
        name=name, prompt=f"What was {name.lower()}?", type="currency"
    )


async def computed(world: World, name: str, op: str, inputs, as_: str | None = None):
    return await world.service.create_column(  # type: ignore[union-attr]
        world.workspace,
        ColumnDraft(
            name=name,
            prompt=f"{name}, computed in code.",
            type="number",
            depends_on=[c.id for c in inputs],
            derivation=Derivation(op=op, **({"as": as_} if as_ else {})),  # type: ignore[arg-type]
        ),
    )


async def fill(world: World, column, reply: str):
    world.script(reply)
    return await world.service.fill_cell(world.workspace, world.acme.id, column.id)  # type: ignore[union-attr]


async def cell_of(world: World, column):
    return await mrepo.get_cell_for(world.pool, world.acme.id, column.id)


# --- the column --------------------------------------------------------------------------------------------------


async def test_a_computed_column_remembers_how_it_is_computed_and_what_it_reads(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )

    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )

    stored = (await mrepo.list_columns(world.pool, world.workspace))[-1]
    assert stored.id == margin.id
    assert stored.derivation == Derivation(op="ratio", **{"as": "percent"})
    assert stored.depends_on == [income.id, revenue.id]


async def test_renaming_a_computed_column_keeps_how_it_is_computed(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )

    renamed = await world.service.update_column(  # type: ignore[union-attr]
        world.workspace, margin.id, ColumnPatch(name="Op margin")
    )

    assert renamed.version == 2 and renamed.derivation == margin.derivation
    assert renamed.depends_on == margin.depends_on


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "text"},  # a computed column is a number or an amount
        {"depends_on": []},  # reads nothing
    ],
)
def test_a_badly_formed_computed_column_is_refused(bad):
    base = {
        "name": "M",
        "prompt": "computed",
        "type": "number",
        "derivation": Derivation(op="ratio", **{"as": "percent"}),  # type: ignore[arg-type]
    }
    from uuid import uuid4

    base["depends_on"] = [uuid4(), uuid4()]
    with pytest.raises(ValueError):
        ColumnDraft(**{**base, **bad})


# --- computing a cell -----------------------------------------------------------------------------------------------


async def test_a_computed_cell_is_exact_traced_and_free(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )
    await fill(world, income, says("$500 million", OPINC))
    await fill(world, revenue, says("$2,000 million", REV))

    result = await world.service.fill_cell(world.workspace, world.acme.id, margin.id)  # type: ignore[union-attr]

    cell = result.cell
    assert cell.status == "answered" and cell.display_value == "25%"
    assert cell.value == {"number": "25", "unit": "%"}
    assert cell.period == "Q2 2026"
    assert (result.cost_usd, result.model_calls) == (
        Decimal(0),
        0,
    )  # arithmetic, not a model call
    assert [e.quote for e in cell.evidence] == [
        OPINC,
        REV,
    ]  # the quotes behind its inputs
    assert cell.derivation is not None and cell.derivation["op"] == "ratio"
    assert [i["column"] for i in cell.derivation["inputs"]] == [
        "Operating income",
        "Revenue",
    ]
    assert cell.prompt_version == "derived-1" and cell.model is None


async def test_a_computed_cell_appears_by_itself_when_its_inputs_are_filled(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )

    await fill(world, income, says("$500 million", OPINC))
    half = await cell_of(world, margin)
    await fill(world, revenue, says("$2,000 million", REV))
    done = await cell_of(world, margin)

    assert (
        half is not None
        and half.status == "not_found"
        and "'Revenue'" in (half.status_detail or "")
    )
    assert (
        done is not None and done.status == "answered" and done.display_value == "25%"
    )


async def test_refilling_an_input_updates_the_computed_cell(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    await computed(world, "Operating margin", "ratio", [income, revenue], "percent")
    await fill(world, income, says("$500 million", OPINC))
    await fill(world, revenue, says("$2,000 million", REV))

    await fill(
        world, income, says("$1,000 million", "x")
    )  # not a real quote: rejected, input unreliable

    margin = (await mrepo.list_columns(world.pool, world.workspace))[-1]
    cell = await cell_of(world, margin)
    # the input is no longer a trustworthy answer, so neither is the margin computed from it
    assert cell is not None and cell.status == "low_confidence" and cell.value is None
    assert "'Operating income' is uncertain" in (cell.status_detail or "")


async def test_a_missing_input_makes_the_computed_cell_not_found_and_names_it(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )

    cell = (
        await world.service.fill_cell(world.workspace, world.acme.id, margin.id)
    ).cell  # type: ignore[union-attr]

    assert cell.status == "not_found" and cell.evidence == [] and cell.value is None
    assert "'Operating income' is not filled yet" in (cell.status_detail or "")


async def test_inputs_from_different_periods_are_not_combined(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )
    await fill(world, income, says("$500 million", OPINC, period="Q2 2026"))
    await fill(world, revenue, says("$2,000 million", REV, period="FY2026"))

    cell = await cell_of(world, margin)

    assert cell is not None and cell.status == "low_confidence" and cell.value is None
    assert "different periods" in (cell.status_detail or "")


async def test_a_period_the_text_never_mentions_is_not_kept(world):
    revenue = await money(world, "Revenue")

    cell = (
        await fill(world, revenue, says("$2,000 million", REV, period="FY2031"))
    ).cell

    assert cell.status == "answered" and cell.period is None


async def test_the_period_the_text_supports_is_kept_in_its_canonical_form(world):
    revenue = await money(world, "Revenue")

    cell = (
        await fill(
            world, revenue, says("$2,000 million", REV, period="second quarter of 2026")
        )
    ).cell

    assert cell.period == "Q2 2026"


async def test_a_cell_a_person_pinned_is_left_alone_by_the_refresh(world):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )
    first = (
        await world.service.fill_cell(world.workspace, world.acme.id, margin.id)
    ).cell  # type: ignore[union-attr]
    await mrepo.set_pinned(world.pool, first.id, True)

    await fill(world, income, says("$500 million", OPINC))
    await fill(world, revenue, says("$2,000 million", REV))

    cell = await cell_of(world, margin)
    assert (
        cell is not None and cell.pinned and cell.status == "not_found"
    )  # not overwritten


# --- tie-outs ----------------------------------------------------------------------------------------------------------


GROSS_PROFIT_CHECK = CheckSpec(
    id="gross_profit_ties",
    name="Gross profit = revenue - cost of revenue",
    kind="difference_identity",
    a="Revenue",
    b="Cost of revenue",
    result="Gross profit",
    tolerance=Decimal("0.005"),
)


async def gross_world(world: World):
    world.service._checks = [GROSS_PROFIT_CHECK]  # type: ignore[union-attr]
    return (
        await money(world, "Revenue"),
        await money(world, "Cost of revenue"),
        await money(world, "Gross profit"),
    )


async def test_numbers_that_tie_show_a_passing_check_on_the_result_cell(world):
    revenue, cost, gross = await gross_world(world)
    await fill(world, revenue, says("$2,000 million", REV))
    await fill(world, cost, says("$800 million", COST))
    await fill(world, gross, says("$1,200 million", GROSS))

    cell = await cell_of(world, gross)

    assert cell is not None and [c["status"] for c in cell.checks] == ["ties"]
    assert cell.checks[0]["id"] == "gross_profit_ties"


async def test_numbers_that_do_not_tie_show_the_gap_and_never_hide_it(world):
    revenue, cost, gross = await gross_world(world)
    await fill(world, revenue, says("$2,000 million", REV))
    await fill(world, cost, says("$800 million", COST))
    await fill(world, gross, says("$1,200 million", GROSS.replace("1,200", "1,200")))
    world.service._checks = [GROSS_PROFIT_CHECK]  # type: ignore[union-attr]
    # Corrupt the stored gross profit to 1,500: the check must now fail.
    await world.pool.execute(
        "update cells set value = jsonb_set(value, '{amount}', '\"1500000000\"') where column_id = $1",
        gross.id,
    )

    await world.service.refresh_row(world.workspace, world.acme.id)  # type: ignore[union-attr]

    cell = await cell_of(world, gross)
    assert cell is not None and [c["status"] for c in cell.checks] == ["does_not_tie"]
    assert "a gap of" in cell.checks[0]["detail"]


async def test_a_check_with_a_missing_input_says_it_could_not_run(world):
    revenue, _, gross = await gross_world(world)
    await fill(world, revenue, says("$2,000 million", REV))
    await fill(
        world, gross, says("$1,200 million", GROSS)
    )  # cost of revenue never filled

    cell = await cell_of(world, gross)

    assert cell is not None and [c["status"] for c in cell.checks] == ["cannot_check"]


async def test_checks_are_not_run_for_columns_the_workspace_does_not_have(world):
    world.service._checks = [GROSS_PROFIT_CHECK]  # type: ignore[union-attr]
    revenue = await money(world, "Revenue")

    cell = (await fill(world, revenue, says("$2,000 million", REV))).cell

    assert cell.checks == []  # no 'Gross profit' column: no check, and not a failure


async def test_a_new_fill_clears_a_stale_check_until_it_is_recomputed(world):
    revenue, cost, gross = await gross_world(world)
    await fill(world, revenue, says("$2,000 million", REV))
    await fill(world, cost, says("$800 million", COST))
    await fill(world, gross, says("$1,200 million", GROSS))

    refreshed = (await fill(world, gross, says("$1,200 million", GROSS))).cell

    assert [c["status"] for c in refreshed.checks] == ["ties"]


# --- a run: ordinary cells first, computed cells after, no model cost for the computed ones ---------------------------------


async def test_a_run_computes_derived_cells_after_their_inputs_and_estimates_only_model_work(
    world,
):
    income, revenue = (
        await money(world, "Operating income"),
        await money(world, "Revenue"),
    )
    margin = await computed(
        world, "Operating margin", "ratio", [income, revenue], "percent"
    )
    settings = world.settings.model_copy(update={"run_concurrency": 1})
    manager = RunManager(world.pool, settings, world.service)  # type: ignore[arg-type]
    world.script(says("$500 million", OPINC), says("$2,000 million", REV))

    plan = await manager.plan(world.workspace, RunScope())
    run = await manager.start(world.workspace, RunScope(), confirmed=True)
    record = await wait_run(manager, world, run.id)

    assert (
        len(plan.cells) == 3 and plan.estimate.cells == 2
    )  # the margin is not counted
    assert record.status == "completed" and record.total == 3 and record.failed == 0
    cell = await cell_of(world, margin)
    assert (
        cell is not None and cell.status == "answered" and cell.display_value == "25%"
    )


# --- found live: "$1.2B" was stored as $1.2, a billion-fold error ------------------------------------------------------------

SCALE_QUOTE = (
    "Q2 2026 adjusted operating income was $1.2B, representing a margin of 62%."
)


async def scaled_world(world: World):
    await world.acme_document(SCALE_QUOTE, name="update.docx")
    return await money(world, "Adjusted operating income")


async def test_a_figure_printed_with_a_B_is_stored_in_billions_not_as_one_dollar_twenty(
    world,
):
    column = await scaled_world(world)

    cell = (await fill(world, column, says("$1.2B", SCALE_QUOTE))).cell

    assert cell.status == "answered"
    assert cell.value["amount"] == "1200000000" and cell.value["scale"] == "billions"
    assert cell.display_value == "$1.2 billion"


async def test_an_answer_that_drops_the_scale_is_sent_back_and_the_fixed_answer_is_kept(
    world,
):
    column = await scaled_world(world)
    world.script(says("$1.2", SCALE_QUOTE), says("$1.2B", SCALE_QUOTE))

    result = await world.service.fill_cell(world.workspace, world.acme.id, column.id)  # type: ignore[union-attr]

    assert result.cell.value["amount"] == "1200000000"  # never $1.2
    assert result.model_calls == 2
    retry_prompt = world.llm.seen[1][1].content
    assert "leaves out" in retry_prompt  # the model was told what was wrong


async def test_an_answer_that_keeps_dropping_the_scale_never_becomes_a_value(world):
    column = await scaled_world(world)

    cell = (
        await fill(world, column, says("$1.2", SCALE_QUOTE))
    ).cell  # the same wrong reply twice

    assert cell.status == "low_confidence" and cell.value is None
    assert "leaves out" in (cell.status_detail or "")


async def test_a_reply_cut_off_mid_json_twice_is_reported_as_cut_off(world):
    column = await money(world, "Revenue")
    cut = '{"status": "answered", "answer": "$2,000 million", "evidence": [{"chunk": "C1", "quo'

    cell = (await fill(world, column, cut)).cell

    assert cell.status == "pending_or_error"
    assert "cut off" in (cell.status_detail or "") and "model call failed" not in (
        cell.status_detail or ""
    )
