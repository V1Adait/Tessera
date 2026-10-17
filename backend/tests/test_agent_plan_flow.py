"""A question becomes columns: the plan card, the confirmation, and what runs on a yes.

A scripted agent model calls propose_columns, a scripted planner proposes columns, real tools and a
real database do the rest. Needs the database (conftest); skips without one.
"""

import json
from decimal import Decimal

import pytest

from backend.app.agent.actions import ActionDesk, ActionNotPendingError
from backend.app.agent.tools import parse_args
from backend.app.costs import Price, PriceBook, set_price_book
from backend.app.db import agent_repository as arepo
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.documents_service import DocumentService
from backend.app.llm_cache import PgLlmCache
from backend.app.models import ConfirmationEvent, ToolEvent
from backend.app.packs.loader import get_packs
from backend.app.planning.planner import ColumnPlanner
from backend.app.tracing import db_recorder
from backend.tests.fakes import QueueFake, make_settings
from backend.tests.test_agent_loop import Rig, conversation, of, tool_results
from backend.tests.test_matrix_flow import World
from backend.tests.test_matrix_runs import wait_run

QUESTION = "Is Acme a good investment?"


def planned(*columns: dict) -> str:
    return json.dumps({"columns": list(columns), "notes": ""})


PACK_COLUMNS = planned(
    {"pack_key": "revenue_growth"},
    {"pack_key": "free_cash_flow"},
    {"pack_key": "total_debt"},
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
    return await World(pool, settings, documents).start()


class PlanRig:
    """An agent turn with a scripted planner, plus the confirmation desk."""

    def __init__(self, world: World, rig: Rig, planner_llm: QueueFake) -> None:
        self.world, self.rig, self.planner_llm = world, rig, planner_llm
        rig.ctx.services.planner = ColumnPlanner(
            planner_llm, provider="fake", model="scripted"
        )
        self.desk = ActionDesk(
            world.pool, rig.settings, rig.ctx.services, rig.deps.registry
        )

    async def ask(self, question: str = QUESTION, **extra_args):
        self.rig.script(
            [("propose_columns", {"question": question, **extra_args})],
            "Please confirm.",
        )
        return await self.rig.run()

    async def pending(self):
        return await arepo.list_actions(
            self.world.pool, self.rig.ctx.conversation_id, pending_only=True
        )

    async def columns(self):
        return [
            c.name
            for c in await mrepo.list_columns(self.world.pool, self.world.workspace)
        ]


@pytest.fixture
async def plan_rig(world, pool) -> PlanRig:
    rig = Rig(world, await conversation(pool, world.workspace))
    await repo.create_entity(
        pool, world.workspace, "Globex"
    )  # two rows: Acme and Globex
    return PlanRig(world, rig, QueueFake(replies=[PACK_COLUMNS]))


# --- what the model is told ---------------------------------------------------------------------------


def test_the_model_sees_only_the_question_and_the_pack_not_the_stored_plan(world, pool):
    from backend.app.agent.tools_impl import build_registry

    schema = next(
        t
        for t in build_registry().schemas()
        if t["function"]["name"] == "propose_columns"
    )["function"]["parameters"]

    assert set(schema["properties"]) == {"question", "pack"}
    assert "$defs" not in schema
    assert schema["required"] == ["question"]


# --- the card: nothing is created before a yes --------------------------------------------------------------------


async def test_a_question_makes_a_plan_card_and_creates_nothing(plan_rig, world, pool):
    events = await plan_rig.ask()

    ask = of(events, ConfirmationEvent)[0]
    assert ask.tool == "propose_columns" and ask.risk == "costly"
    assert ask.plan is not None
    assert [c["name"] for c in ask.plan["columns"]] == [
        "Revenue growth",
        "Free cash flow",
        "Total debt",
    ]
    assert ask.plan["rows"] == 2 and ask.plan["cells"] == 6
    assert "Add 3 column(s) and fill 6 cells for 2 companies" in ask.summary
    assert await plan_rig.columns() == []  # nothing yet
    assert await pool.fetchval("select count(*) from matrix_runs") == 0


async def test_the_card_carries_the_columns_types_and_where_each_came_from(plan_rig):
    plan_rig.planner_llm.replies[:] = [
        planned(
            {"pack_key": "revenue"},
            {
                "name": "Backlog",
                "prompt": "What is the order backlog?",
                "type": "currency",
            },
        )
    ]

    ask = of(await plan_rig.ask("revenue and backlog"), ConfirmationEvent)[0]

    assert [(c["name"], c["type"], c["source"]) for c in ask.plan["columns"]] == [
        ("Revenue", "currency", "pack"),
        ("Backlog", "currency", "new"),
    ]


async def test_the_card_shows_a_cost_estimate_from_past_calls_or_prices(
    plan_rig, world
):
    set_price_book(
        PriceBook(
            {
                world.settings.extract_model: Price(
                    Decimal("0.000001"), Decimal("0.000002")
                )
            }
        )
    )

    ask = of(await plan_rig.ask(), ConfirmationEvent)[0]

    assert ask.plan["estimate_usd"] is not None and ask.plan["basis"] == "price_list"
    assert "about $" in ask.summary


async def test_an_unknown_cost_is_said_to_be_unknown_not_shown_as_free(plan_rig):
    ask = of(await plan_rig.ask(), ConfirmationEvent)[0]

    assert ask.plan["estimate_usd"] is None and "unknown cost" in ask.summary


async def test_the_plan_is_stored_for_a_yes_with_exactly_what_the_card_showed(plan_rig):
    await plan_rig.ask()

    (action,) = await plan_rig.pending()

    assert action.tool == "propose_columns" and action.risk == "costly"
    assert [c["name"] for c in action.args["columns"]] == [
        "Revenue growth",
        "Free cash flow",
        "Total debt",
    ]
    assert (
        action.args["plan"]["cells"] == 6
    )  # the card can be shown again after a reload


async def test_columns_typed_by_the_model_itself_are_ignored(plan_rig):
    """The model passes its own columns; the stored plan still comes from the planner and code."""
    sneaky = [
        {
            "name": "Hacked",
            "prompt": "x?",
            "type": "text",
            "source": "new",
            "allowed_values": [],
            "applicability": None,
            "evidence_required": False,
        }
    ]

    await plan_rig.ask(columns=sneaky)

    (action,) = await plan_rig.pending()
    assert "Hacked" not in json.dumps(action.args)


# --- nothing to confirm -----------------------------------------------------------------------------------------------


async def test_a_failing_planner_creates_no_card_and_tells_the_model_why(
    plan_rig, pool
):
    plan_rig.planner_llm.replies[:] = [RuntimeError("provider down")]

    events = await plan_rig.ask()

    assert not of(events, ConfirmationEvent)
    assert "could not turn that into columns" in tool_results(plan_rig.rig.llm, 1)[0]
    assert await plan_rig.pending() == []


async def test_a_plan_with_no_valid_columns_creates_no_card(plan_rig):
    plan_rig.planner_llm.replies[:] = [planned({"name": "", "prompt": ""})]

    events = await plan_rig.ask()

    assert not of(events, ConfirmationEvent)
    assert "specific facts" in tool_results(plan_rig.rig.llm, 1)[0]


async def test_an_unknown_pack_is_reported_not_guessed(plan_rig):
    events = await plan_rig.ask(pack="no_such_pack")

    assert not of(events, ConfirmationEvent)
    assert "no pack 'no_such_pack'" in tool_results(plan_rig.rig.llm, 1)[0]


async def test_without_a_planner_the_tool_says_so(world, pool):
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script([("propose_columns", {"question": QUESTION})], "ok")

    events = await rig.run()

    assert not of(events, ConfirmationEvent)
    assert "unavailable" in tool_results(rig.llm, 1)[0]


# --- the yes ------------------------------------------------------------------------------------------------------------


async def test_confirming_creates_the_columns_and_starts_filling_them(
    plan_rig, world, pool
):
    await plan_rig.ask()
    (action,) = await plan_rig.pending()

    outcome = await plan_rig.desk.confirm(action.id)

    assert outcome.ok and outcome.changed_matrix
    assert await plan_rig.columns() == [
        "Revenue growth",
        "Free cash flow",
        "Total debt",
    ]
    assert outcome.data["question"] == QUESTION
    run = await wait_run(plan_rig.rig.ctx.services.runs, world, outcome.data["run_id"])
    assert run.total == 6 and run.status == "completed"


async def test_a_second_click_cannot_run_the_plan_twice(plan_rig, world):
    await plan_rig.ask()
    (action,) = await plan_rig.pending()
    outcome = await plan_rig.desk.confirm(action.id)
    await wait_run(plan_rig.rig.ctx.services.runs, world, outcome.data["run_id"])

    with pytest.raises(ActionNotPendingError):
        await plan_rig.desk.confirm(action.id)

    assert len(await plan_rig.columns()) == 3


async def test_cancelling_creates_nothing(plan_rig):
    await plan_rig.ask()
    (action,) = await plan_rig.pending()

    await plan_rig.desk.cancel(action.id)

    assert await plan_rig.columns() == []


async def test_the_tool_refuses_to_create_anything_without_a_confirmation(
    plan_rig, world
):
    """A model that passes the stored plan itself still cannot skip the card."""
    await plan_rig.ask()
    (action,) = await plan_rig.pending()
    spec = plan_rig.rig.deps.registry.get("propose_columns")
    assert spec is not None
    args = parse_args(spec, action.args)

    result = await spec.run(plan_rig.rig.ctx, args)  # ctx.confirmed is False here

    assert not result.ok and "confirmed by the person" in result.text
    assert await plan_rig.columns() == []


async def test_a_reused_column_is_not_created_twice_and_only_its_missing_cells_are_filled(
    plan_rig, world
):
    await world.column(
        name="Revenue growth", prompt="By what percent did revenue grow?"
    )
    await plan_rig.ask()
    (action,) = await plan_rig.pending()

    outcome = await plan_rig.desk.confirm(action.id)

    assert (await plan_rig.columns()).count("Revenue growth") == 1
    assert "reused 1" in outcome.message
    run = await wait_run(plan_rig.rig.ctx.services.runs, world, outcome.data["run_id"])
    assert run.total == 6  # 2 new columns x 2 rows + the reused column's 2 empty cells


async def test_with_no_rows_the_columns_are_created_and_nothing_is_filled(world, pool):
    await pool.execute("delete from entities")
    rig = Rig(world, await conversation(pool, world.workspace))
    p = PlanRig(world, rig, QueueFake(replies=[PACK_COLUMNS]))
    events = await p.ask()
    ask = of(events, ConfirmationEvent)[0]
    assert "no companies yet" in ask.summary and ask.plan["cells"] == 0

    (action,) = await p.pending()
    outcome = await p.desk.confirm(action.id)

    assert outcome.ok and "nothing was filled" in outcome.message
    assert "run_id" not in outcome.data and len(await p.columns()) == 3


async def test_a_run_already_going_does_not_stop_the_columns_being_created(
    plan_rig, world, pool
):
    await plan_rig.ask()
    (action,) = await plan_rig.pending()
    await pool.execute(
        "insert into matrix_runs (workspace_id, status, scope, total, cap_usd) "
        "values ($1, 'running', '{}', 1, 1)",
        world.workspace,
    )

    outcome = await plan_rig.desk.confirm(action.id)

    assert outcome.ok and "already going" in outcome.message
    assert len(await plan_rig.columns()) == 3


async def test_a_column_made_by_hand_after_the_plan_was_shown_is_used_not_duplicated(
    plan_rig, world
):
    await plan_rig.ask()
    (action,) = await plan_rig.pending()
    await world.column(
        name="Free cash flow", prompt="What was free cash flow?"
    )  # appears meanwhile

    outcome = await plan_rig.desk.confirm(action.id)

    assert outcome.ok
    assert (await plan_rig.columns()).count("Free cash flow") == 1


async def test_the_confirmation_is_recorded_in_the_conversation(plan_rig, world, pool):
    await plan_rig.ask()
    (action,) = await plan_rig.pending()
    outcome = await plan_rig.desk.confirm(action.id)
    await wait_run(plan_rig.rig.ctx.services.runs, world, outcome.data["run_id"])

    last = (await repo.list_messages(pool, plan_rig.rig.ctx.conversation_id))[-1]

    assert last.role == "assistant" and "Added 3 column(s)" in last.content


async def test_the_planner_call_is_traced_and_an_identical_question_is_free(
    plan_rig, pool
):
    plan_rig.rig.ctx.services.planner = ColumnPlanner(
        plan_rig.planner_llm,
        provider="fake",
        model="scripted",
        recorder=db_recorder(pool),
        cache=PgLlmCache(pool),
    )

    await plan_rig.ask()
    await (
        plan_rig.ask()
    )  # the same question: answered from the cache, no second model call

    assert await pool.fetchval("select count(*) from traces where kind = 'plan'") == 1
    assert len(plan_rig.planner_llm.seen) == 1


# --- packs --------------------------------------------------------------------------------------------------------------


async def test_list_packs_names_the_shipped_pack(world, pool):
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script([("list_packs", {})], "ok")

    await rig.run()

    assert "credit_pe_diligence" in tool_results(rig.llm, 1)[0]


async def test_add_pack_adds_every_column_and_skips_existing_ones(world, pool):
    await world.column(name="Revenue", prompt="What was revenue?")
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script([("add_pack", {"pack": "credit_pe_diligence"})], "ok")

    events = await rig.run()

    names = [c.name for c in await mrepo.list_columns(pool, world.workspace)]
    assert len(names) == len(get_packs()["credit_pe_diligence"].columns)
    assert names.count("Revenue") == 1
    assert of(events, ToolEvent)[-1].changed_matrix is True


async def test_add_pack_can_take_only_some_columns(world, pool):
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script(
        [("add_pack", {"pack": "credit_pe_diligence", "keys": ["revenue", "cash"]})],
        "ok",
    )

    await rig.run()

    assert [c.name for c in await mrepo.list_columns(pool, world.workspace)] == [
        "Revenue",
        "Cash and equivalents",
    ]


async def test_add_pack_rejects_an_unknown_pack_or_key(world, pool):
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script([("add_pack", {"pack": "nope"})], "ok")
    await rig.run()
    assert "no pack 'nope'" in tool_results(rig.llm, 1)[0]

    rig.llm.seen.clear()  # so the next assertions read THIS run's prompts
    rig.script(
        [("add_pack", {"pack": "credit_pe_diligence", "keys": ["revenue", "bogus"]})],
        "ok",
    )
    await rig.run()
    assert "bogus" in tool_results(rig.llm, 1)[0]
    assert await mrepo.list_columns(pool, world.workspace) == []  # nothing half-added


# --- computed columns -----------------------------------------------------------------------------------------------------


async def test_a_question_that_needs_a_margin_plans_the_computed_column_and_its_inputs(
    plan_rig,
):
    plan_rig.planner_llm.replies[:] = [planned({"pack_key": "operating_margin"})]

    ask = of(await plan_rig.ask("What is the operating margin?"), ConfirmationEvent)[0]

    assert [(c["name"], c["derived"] is not None) for c in ask.plan["columns"]] == [
        ("Operating income", False),
        ("Revenue", False),
        ("Operating margin", True),
    ]
    assert (
        ask.plan["cells"] == 4
    )  # 2 rows x the 2 columns the model must read: the margin is free
    assert "1 of them are computed in code, at no cost" in ask.summary


async def test_confirming_creates_the_computed_column_linked_to_the_columns_it_reads(
    plan_rig, world
):
    plan_rig.planner_llm.replies[:] = [planned({"pack_key": "operating_margin"})]
    await plan_rig.ask("What is the operating margin?")
    (action,) = await plan_rig.pending()

    outcome = await plan_rig.desk.confirm(action.id)

    columns = {c.name: c for c in await mrepo.list_columns(world.pool, world.workspace)}
    margin = columns["Operating margin"]
    assert margin.derivation is not None and margin.derivation.op == "ratio"
    assert margin.depends_on == [columns["Operating income"].id, columns["Revenue"].id]
    run = await wait_run(plan_rig.rig.ctx.services.runs, world, outcome.data["run_id"])
    assert (
        run.total == 6 and run.status == "completed"
    )  # 3 columns x 2 rows, computed ones included


async def test_a_stored_plan_with_a_computed_column_survives_being_read_back(plan_rig):
    plan_rig.planner_llm.replies[:] = [planned({"pack_key": "net_debt_to_ebitda"})]
    await plan_rig.ask("How leveraged is it?")

    (action,) = await plan_rig.pending()

    names = [c["name"] for c in action.args["columns"]]
    assert names == [
        "Total debt",
        "Cash and equivalents",
        "Adjusted EBITDA",
        "Net debt to EBITDA",
    ]
    assert action.args["columns"][-1]["inputs"] == names[:3]


async def test_add_pack_links_a_computed_column_to_what_it_reads(world, pool):
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script(
        [("add_pack", {"pack": "credit_pe_diligence", "keys": ["net_debt"]})], "ok"
    )

    await rig.run()

    columns = {c.name: c for c in await mrepo.list_columns(pool, world.workspace)}
    assert set(columns) == {
        "Total debt",
        "Cash and equivalents",
        "Net debt",
    }  # its inputs came with it
    net_debt = columns["Net debt"]
    assert net_debt.derivation is not None and net_debt.type == "currency"
    assert net_debt.depends_on == [
        columns["Total debt"].id,
        columns["Cash and equivalents"].id,
    ]


async def test_add_pack_creates_every_computed_column_of_the_pack(world, pool):
    rig = Rig(world, await conversation(pool, world.workspace))
    rig.script([("add_pack", {"pack": "credit_pe_diligence"})], "ok")

    await rig.run()

    computed = [
        c.name for c in await mrepo.list_columns(pool, world.workspace) if c.derivation
    ]
    assert {
        "Operating margin",
        "Net debt",
        "Net debt to EBITDA",
        "Interest coverage",
    } <= set(computed)
