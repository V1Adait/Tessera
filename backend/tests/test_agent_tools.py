"""The tools one by one, against a real database. No model is involved: a tool is just a function
that takes validated arguments, so each rule can be checked directly.
"""

import pytest

from backend.app.agent.tools import Services, ToolContext, parse_args
from backend.app.agent.tools_impl import build_registry
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.documents_service import DocumentService
from backend.app.matrix.models import ColumnDraft
from backend.app.matrix.runs import RunManager
from backend.tests.fakes import make_settings
from backend.tests.test_matrix_flow import World, good

REGISTRY = build_registry()


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


@pytest.fixture
async def ctx(world, pool) -> ToolContext:
    conversation = await repo.create_conversation(pool, "t", world.workspace)
    return ToolContext(
        pool=pool,
        settings=world.settings,
        services=Services(
            matrix=world.service,
            runs=RunManager(pool, world.settings, world.service),  # type: ignore[arg-type]
            documents=world.documents,
        ),
        workspace_id=world.workspace,
        conversation_id=conversation,
    )


async def call(ctx, tool, /, **args):
    spec = REGISTRY.get(tool)
    assert spec is not None
    return await spec.run(ctx, parse_args(spec, args))


async def gate(ctx, tool, /, **args):
    spec = REGISTRY.get(tool)
    assert spec is not None and spec.gate is not None
    return await spec.gate(ctx, parse_args(spec, args))


# --- the tool list -------------------------------------------------------------------------------------


def test_the_schemas_are_plain_json_the_model_can_read():
    schemas = REGISTRY.schemas()
    by_name = {s["function"]["name"]: s["function"] for s in schemas}

    assert len(by_name) == len(schemas)  # no duplicates
    assert by_name["add_column"]["parameters"]["required"] == ["name", "prompt"]
    assert by_name["add_column"]["parameters"]["properties"]["type"]["enum"]
    assert all(f["description"] for f in by_name.values())


def test_only_dangerous_tools_have_a_gate():
    gated = {n for n in REGISTRY.names() if (REGISTRY.get(n) or object()).gate}  # type: ignore[union-attr]
    # propose_columns spends money (it fills what it creates), so it waits for a yes too
    assert gated == {"fill_cells", "remove_row", "remove_column", "propose_columns"}


# --- names: exact, near, ambiguous, unknown ---------------------------------------------------------------


async def test_a_row_is_found_by_name_ignoring_case(ctx, world):
    await world.column()
    result = await call(ctx, "get_cell", row="aCmE", column="employees")

    assert result.ok and "not been filled" in result.text


async def test_an_unknown_name_lists_the_names_that_exist(ctx):
    result = await call(ctx, "get_cell", row="Initech", column="x")

    assert not result.ok
    assert "no row called 'Initech'" in result.text and "'Acme'" in result.text


async def test_a_name_that_could_mean_two_rows_is_not_guessed(ctx, world, pool):
    await repo.create_entity(pool, world.workspace, "Acme Europe")
    await repo.create_entity(pool, world.workspace, "Acme Asia")

    result = await call(
        ctx, "remove_row", row="Acme E"
    )  # near: only one contains "acme e"
    assert result.ok  # a single near match is accepted

    result = await call(ctx, "get_cell", row="Acm", column="x")
    assert not result.ok and "More than one row" in result.text


# --- reading ---------------------------------------------------------------------------------------------------


async def test_list_matrix_shows_rows_columns_cells_and_documents(ctx, world):
    await world.acme_document()
    await world.column()

    text = (await call(ctx, "list_matrix")).text

    assert "Rows (1): Acme" in text
    assert "Employees (number)" in text
    assert "report.docx" in text and "row 'Acme'" in text


async def test_get_cell_returns_the_value_and_numbered_quotes(ctx, world):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await world.fill(column)

    result = await call(ctx, "get_cell", row="Acme", column="Employees")

    assert "166,000" in result.text and "answered" in result.text
    assert (
        '[S1] report.docx: "approximately 166,000 full-time equivalent employees"'
        in result.text
    )
    assert ctx.sources["S1"].filename == "report.docx"


async def test_search_numbers_passages_continuing_across_searches(ctx, world):
    await world.acme_document()

    first = await call(ctx, "search_documents", query="employees")
    after_first = set(ctx.sources)
    second = await call(ctx, "search_documents", query="employees full-time")

    assert "\n[S1] " in first.text and after_first == {"S1"}
    # numbering continues, never restarts
    assert "\n[S2] " in second.text and "\n[S1] " not in second.text


async def test_a_search_with_no_match_tells_the_model_not_to_answer_from_memory(ctx):
    result = await call(ctx, "search_documents", query="zebra quantum pizza")

    assert result.ok and "do not answer from memory" in result.text


async def test_a_search_can_be_limited_to_one_row(ctx, world, pool):
    globex = await repo.create_entity(pool, world.workspace, "Globex")
    await world.acme_document()

    mine = await call(ctx, "search_documents", query="employees", row="Globex")

    assert globex and "No passage" in mine.text


# --- changing the matrix -----------------------------------------------------------------------------------------


async def test_add_row_rejects_a_duplicate_name(ctx):
    assert (await call(ctx, "add_row", name="Globex")).ok
    again = await call(ctx, "add_row", name="globex")

    assert not again.ok and "already exists" in again.text


async def test_add_column_validates_like_the_api_does(ctx):
    bad = await call(ctx, "add_column", name="Region", prompt="Where?", type="select")

    assert not bad.ok and "at least two allowed values" in bad.text


async def test_add_and_edit_a_column(ctx, world):
    added = await call(
        ctx,
        "add_column",
        name="Region",
        prompt="Which region?",
        type="select",
        allowed_values=["EU", "US"],
    )
    assert added.ok and added.changed_matrix

    edited = await call(
        ctx, "edit_column", column="Region", prompt="Which sales region leads?"
    )
    assert edited.ok and "out of date" in edited.text
    columns = await mrepo.list_columns(ctx.pool, ctx.workspace_id)
    assert columns[0].prompt == "Which sales region leads?" and columns[0].version == 2


async def test_edit_column_with_nothing_to_change_says_so(ctx, world):
    await world.column()

    result = await call(ctx, "edit_column", column="Employees")

    assert not result.ok and "Nothing to change" in result.text


async def test_assign_document_moves_it_to_a_row(ctx, world, pool):
    globex = await repo.create_entity(pool, world.workspace, "Globex")
    document = await world.acme_document()

    result = await call(ctx, "assign_document", document="report", row="Globex")

    assert result.ok
    moved = await repo.get_document(pool, document.id)
    assert moved is not None and moved.entity_id == globex.id


# --- removing --------------------------------------------------------------------------------------------------------


async def test_the_removal_gate_counts_what_would_be_lost(ctx, world):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await world.fill(column)

    asked = await gate(ctx, "remove_row", row="Acme")

    assert asked is not None and asked.risk == "destructive"
    assert "1 cells (1 answered)" in asked.summary and "documents stay" in asked.summary


async def test_removing_a_row_deletes_its_cells_but_keeps_its_documents(
    ctx, world, pool
):
    document = await world.acme_document()
    column = await world.column()
    world.script(good())
    await world.fill(column)

    result = await call(ctx, "remove_row", row="Acme")

    assert result.ok and result.changed_matrix
    assert await repo.list_entities(pool, world.workspace) == []
    assert await mrepo.list_cells(pool, world.workspace) == []
    kept = await repo.get_document(pool, document.id)
    assert kept is not None and kept.entity_id is None


async def test_a_column_other_columns_depend_on_cannot_be_removed(ctx, world):
    base = await world.column(name="Base")
    await world.service.create_column(  # type: ignore[union-attr]
        world.workspace,
        ColumnDraft(name="Derived", prompt="Derived from base", depends_on=[base.id]),
    )

    result = await call(ctx, "remove_column", column="Base")

    assert not result.ok and "depend on 'Base': Derived" in result.text


async def test_nothing_is_removed_while_a_run_is_going(ctx, world, pool):
    await pool.execute(
        "insert into matrix_runs (workspace_id, status, scope, total, cap_usd) "
        "values ($1, 'running', '{}'::jsonb, 1, 1)",
        world.workspace,
    )

    result = await call(ctx, "remove_row", row="Acme")

    assert not result.ok and "run is going" in result.text
    assert len(await repo.list_entities(pool, world.workspace)) == 1


# --- filling ----------------------------------------------------------------------------------------------------------


async def test_plan_fill_spends_nothing_and_reports_the_scope(ctx, world, pool):
    await world.acme_document()
    await world.column()

    result = await call(ctx, "plan_fill", rows=["Acme"])

    assert "1 cells would be filled" in result.text
    assert await pool.fetchval("select count(*) from matrix_runs") == 0
    assert world.llm.seen == []


async def test_filling_an_unknown_row_is_explained_not_run(ctx, world):
    await world.column()

    result = await call(ctx, "fill_cells", rows=["Nobody"])

    assert not result.ok and "no row called 'Nobody'" in result.text


async def test_nothing_to_fill_is_not_an_error(ctx, world):
    result = await call(ctx, "fill_cells")

    assert result.ok and "Nothing needs filling" in result.text


async def test_stop_run_with_no_run_going_is_harmless(ctx):
    result = await call(ctx, "stop_run")

    assert result.ok and "No run is going" in result.text
