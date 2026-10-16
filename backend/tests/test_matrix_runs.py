"""Filling many cells at once: what a run plans, what it costs, how it ends.

The model is scripted, so every outcome is exact. Needs the database (conftest).
"""

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest

from backend.app.api_runs import get_runs
from backend.app.costs import Price, PriceBook, set_price_book
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db import runs_repository as rrepo
from backend.app.db.pool import get_pool
from backend.app.db.records import TraceRecord
from backend.app.documents_service import DocumentService
from backend.app.main import app
from backend.app.matrix.models import ColumnPatch
from backend.app.matrix.runs import (
    ConfirmationRequired,
    InvalidCap,
    RunInProgressError,
    RunManager,
    RunScope,
    ScopeError,
    choose_cells,
)
from backend.tests.fakes import make_settings
from backend.tests.test_matrix_flow import World, good

D = Decimal


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


def manager_for(world: World, **overrides) -> RunManager:
    settings = world.settings.model_copy(update={"run_concurrency": 1, **overrides})
    return RunManager(world.pool, settings, world.service)  # type: ignore[arg-type]


async def make_columns(world: World, n: int):
    return [
        await world.column(
            name=f"Col{i}", prompt=f"How many people work at the company? ({i})"
        )
        for i in range(n)
    ]


async def wait_run(manager: RunManager, world: World, run_id, timeout=30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        record = await manager.get(world.workspace, run_id)
        assert record is not None
        if record.status != "running":
            return record
        await asyncio.sleep(0.03)
    raise AssertionError("the run never finished")


# --- choosing the cells (pure) ----------------------------------------------------------------

E1, E2, C1, C2 = uuid4(), uuid4(), uuid4(), uuid4()


def ids(cells):
    return [(c.entity_id, c.column_id, c.reason) for c in cells]


def test_with_nothing_filled_every_cell_is_missing_rows_first_then_columns():
    cells, pinned = choose_cells([E1, E2], [C1, C2], {}, RunScope())

    assert ids(cells) == [
        (E1, C1, "missing"),
        (E1, C2, "missing"),
        (E2, C1, "missing"),
        (E2, C2, "missing"),
    ]
    assert pinned == 0


def test_answered_cells_are_left_alone_by_default():
    states = {
        (E1, C1): ("answered", False, False),
        (E1, C2): ("not_found", False, False),
    }

    cells, _ = choose_cells([E1], [C1, C2], states, RunScope())

    assert cells == []  # not_found is a result, not a failure


def test_stale_cells_and_error_cells_are_redone():
    states = {
        (E1, C1): ("answered", True, False),
        (E1, C2): ("pending_or_error", False, False),
    }

    cells, _ = choose_cells([E1], [C1, C2], states, RunScope())

    assert ids(cells) == [(E1, C1, "stale"), (E1, C2, "pending_or_error")]


def test_other_statuses_are_redone_only_when_asked():
    states = {
        (E1, C1): ("not_found", False, False),
        (E1, C2): ("low_confidence", False, False),
    }

    cells, _ = choose_cells(
        [E1], [C1, C2], states, RunScope(statuses=["not_found", "low_confidence"])
    )

    assert ids(cells) == [(E1, C1, "not_found"), (E1, C2, "low_confidence")]


def test_missing_and_stale_can_each_be_switched_off():
    states = {(E1, C1): ("answered", True, False)}

    cells, _ = choose_cells(
        [E1], [C1, C2], states, RunScope(missing=False, stale=False)
    )

    assert cells == []


def test_a_pinned_cell_is_never_touched_unless_asked_and_is_counted():
    states = {
        (E1, C1): ("pending_or_error", False, True),
        (E1, C2): ("answered", True, True),
    }

    cells, pinned = choose_cells([E1], [C1, C2], states, RunScope())
    forced, none_pinned = choose_cells(
        [E1], [C1, C2], states, RunScope(include_pinned=True)
    )

    assert cells == [] and pinned == 2
    assert len(forced) == 2 and none_pinned == 0


def test_an_empty_list_of_ids_is_refused_not_read_as_all():
    with pytest.raises(ValueError):
        RunScope(entity_ids=[])
    with pytest.raises(ValueError):
        RunScope(column_ids=[])


# --- planning and the estimate (database) ------------------------------------------------------


async def test_a_plan_lists_the_cells_and_spends_nothing(world):
    await make_columns(world, 3)
    manager = manager_for(world)
    world.script(good())

    plan = await manager.plan(world.workspace, RunScope())

    assert len(plan.cells) == 3 and plan.by_reason == {"missing": 3}
    assert world.llm.seen == []


async def test_a_plan_can_be_narrowed_to_one_row_or_some_columns(world):
    beta = await repo.create_entity(world.pool, world.workspace, "Beta")
    columns = await make_columns(world, 3)
    manager = manager_for(world)

    one_row = await manager.plan(world.workspace, RunScope(entity_ids=[beta.id]))
    some_columns = await manager.plan(
        world.workspace, RunScope(column_ids=[columns[0].id, columns[2].id])
    )

    assert {c.entity_id for c in one_row.cells} == {beta.id} and len(one_row.cells) == 3
    assert len(some_columns.cells) == 4  # 2 rows x 2 columns


async def test_naming_a_row_or_column_from_elsewhere_is_refused(world):
    other = await repo.create_workspace(world.pool, "Other")
    stranger = await repo.create_entity(world.pool, other.id, "Stranger")
    await make_columns(world, 1)
    manager = manager_for(world)

    with pytest.raises(ScopeError):
        await manager.plan(world.workspace, RunScope(entity_ids=[stranger.id]))
    with pytest.raises(ScopeError):
        await manager.plan(world.workspace, RunScope(column_ids=[uuid4()]))
    with pytest.raises(repo.WorkspaceNotFoundError):
        await manager.plan(uuid4(), RunScope())


@pytest.mark.parametrize("cap", ["0", "-1", "1000"])
async def test_a_cap_must_be_above_zero_and_within_the_ceiling(world, cap):
    with pytest.raises(InvalidCap):
        await manager_for(world).plan(world.workspace, RunScope(), D(cap))


async def test_with_no_history_and_no_price_the_cost_is_unknown_and_needs_a_yes(world):
    await make_columns(world, 1)

    plan = await manager_for(world).plan(world.workspace, RunScope())

    assert plan.estimate.basis == "unknown" and plan.estimate.expected_usd is None
    assert plan.needs_confirmation and "cannot be estimated" in (
        plan.confirmation_reason or ""
    )


async def test_a_published_price_gives_an_estimate_when_there_is_no_history(world):
    set_price_book(
        PriceBook({world.settings.extract_model: Price(D("0.0000001"), D("0.0000005"))})
    )
    await make_columns(world, 4)

    plan = await manager_for(world).plan(world.workspace, RunScope())

    per_call = D("0.0000001") * 7000 + D("0.0000005") * 200
    assert plan.estimate.basis == "price_list"
    assert plan.estimate.per_cell_usd == per_call * D("1.3")
    assert plan.estimate.expected_usd == per_call * D("1.3") * 4
    assert not plan.needs_confirmation  # a few tenths of a cent


async def record_calls(world: World, n: int, cost: str):
    for _ in range(n):
        await repo.record_trace(
            world.pool,
            TraceRecord(
                request_id=uuid4(),
                kind="extract",
                status="ok",
                model=world.settings.extract_model,
                cost_usd=D(cost),
                cost_source="provider",
            ),
        )


async def test_what_past_calls_really_cost_beats_the_published_price(world):
    set_price_book(
        PriceBook({world.settings.extract_model: Price(D(1), D(1))})
    )  # absurd on purpose
    await record_calls(world, 6, "0.001")
    await make_columns(world, 10)

    plan = await manager_for(world).plan(world.workspace, RunScope())

    assert plan.estimate.basis == "measured" and plan.estimate.samples == 6
    assert plan.estimate.per_cell_usd == D(
        "0.0013"
    ) and plan.estimate.expected_usd == D("0.013")


async def test_too_few_past_calls_are_not_trusted_as_an_average(world):
    await record_calls(world, 2, "0.5")
    await make_columns(world, 1)

    plan = await manager_for(world).plan(world.workspace, RunScope())

    assert plan.estimate.basis == "unknown"


async def test_a_run_needs_a_yes_when_it_is_dear_or_big_but_not_otherwise(world):
    await record_calls(world, 6, "0.01")
    await make_columns(world, 5)  # 5 cells at 0.013 each = 0.065

    dear = await manager_for(world).plan(world.workspace, RunScope())
    cheap = await manager_for(world, run_confirm_above_usd=D("1")).plan(
        world.workspace, RunScope()
    )
    big = await manager_for(
        world, run_confirm_above_usd=D("1"), run_confirm_above_cells=4
    ).plan(world.workspace, RunScope())

    assert dear.needs_confirmation and "estimated" in (dear.confirmation_reason or "")
    assert not cheap.needs_confirmation
    assert big.needs_confirmation and "5 cells" in (big.confirmation_reason or "")


async def test_a_plan_with_nothing_to_do_needs_no_yes(world):
    plan = await manager_for(world).plan(world.workspace, RunScope())

    assert plan.cells == [] and not plan.needs_confirmation


async def test_the_plan_warns_when_it_will_hit_its_cap_and_when_cells_are_pinned(world):
    await record_calls(world, 6, "0.1")
    columns = await make_columns(world, 3)
    await world.acme_document()
    world.script(good())
    cell = await world.fill(columns[0])
    await mrepo.set_pinned(world.pool, cell.id, True)
    await world.pool.execute(
        "update cells set status = 'pending_or_error', value = null where id = $1",
        cell.id,
    )

    plan = await manager_for(world).plan(world.workspace, RunScope(), D("0.01"))

    assert any("above the cap" in w for w in plan.warnings)
    assert any("1 cell(s) set by a person" in w for w in plan.warnings)


# --- running -----------------------------------------------------------------------------------


async def test_a_run_fills_exactly_the_planned_cells_and_keeps_the_books(world):
    beta = await repo.create_entity(world.pool, world.workspace, "Beta")  # no documents
    await world.acme_document()
    await make_columns(world, 2)
    world.llm.reported_cost = 0.0002
    world.script(good())
    manager = manager_for(world)

    started = await manager.start(world.workspace, RunScope(), confirmed=True)
    record = await wait_run(manager, world, started.id)

    assert (
        record.status,
        record.total,
        record.done,
        record.failed,
        record.skipped,
    ) == ("completed", 4, 4, 0, 0)
    assert record.spent_usd == D("0.0004")  # only acme's two cells called the model
    cells = await mrepo.list_cells(world.pool, world.workspace)
    by_entity = {(c.entity_id, c.status) for c in cells}
    assert by_entity == {(world.acme.id, "answered"), (beta.id, "not_found")}
    assert record.finished_at is not None


async def test_a_run_scoped_to_one_row_touches_only_that_row(world):
    beta = await repo.create_entity(world.pool, world.workspace, "Beta")
    await world.acme_document()
    await make_columns(world, 2)
    world.script(good())
    manager = manager_for(world)

    run = await manager.start(
        world.workspace, RunScope(entity_ids=[world.acme.id]), confirmed=True
    )
    await wait_run(manager, world, run.id)

    assert {
        c.entity_id for c in await mrepo.list_cells(world.pool, world.workspace)
    } == {world.acme.id}
    assert beta.id not in {
        c.entity_id for c in await mrepo.list_cells(world.pool, world.workspace)
    }


async def test_running_again_does_nothing_when_nothing_needs_doing(world):
    await world.acme_document()
    await make_columns(world, 2)
    world.script(good())
    manager = manager_for(world)
    first = await manager.start(world.workspace, RunScope(), confirmed=True)
    await wait_run(manager, world, first.id)
    world.llm.seen.clear()

    second = await manager.start(world.workspace, RunScope())

    assert (second.status, second.total) == ("completed", 0) and world.llm.seen == []


async def test_a_changed_column_is_the_only_thing_redone(world):
    await world.acme_document()
    columns = await make_columns(world, 2)
    world.script(good())
    manager = manager_for(world)
    await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )
    await world.service.update_column(
        world.workspace,
        columns[0].id,
        ColumnPatch(prompt="How many employees work at the company at year end?"),
    )  # type: ignore[union-attr]
    world.llm.seen.clear()

    run = await manager.start(world.workspace, RunScope(), confirmed=True)
    record = await wait_run(manager, world, run.id)

    assert record.total == 1 and len(world.llm.seen) == 1


async def test_a_cell_that_ended_in_error_is_picked_up_by_the_next_run(world):
    await world.acme_document()
    await make_columns(world, 1)
    manager = manager_for(world)
    world.script(RuntimeError("provider down"))
    first = await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )
    world.script(good())

    second = await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )

    assert first.failed == 1
    assert second.total == 1 and second.failed == 0
    assert (await mrepo.list_cells(world.pool, world.workspace))[0].status == "answered"


# --- the spending cap, stopping, concurrency, failure ----------------------------------------


async def test_a_run_stops_by_itself_at_its_cap_and_says_so(world):
    await world.acme_document()
    await make_columns(world, 8)
    world.llm.reported_cost = 0.001
    world.script(good())
    manager = manager_for(world, run_concurrency=1)

    run = await manager.start(
        world.workspace, RunScope(), confirmed=True, cap_usd=D("0.0025")
    )
    record = await wait_run(manager, world, run.id)

    assert record.status == "cost_cap"
    assert (record.done, record.skipped) == (
        3,
        5,
    )  # the third crossed the cap; nothing after it began
    assert record.spent_usd == D("0.003") and "cap" in (record.detail or "")


async def test_a_person_can_stop_a_run_and_the_cells_in_flight_finish(world):
    await world.acme_document()
    await make_columns(world, 6)
    world.script(good())
    gate, entered = asyncio.Event(), asyncio.Event()
    real = world.service.fill_cell  # type: ignore[union-attr]

    async def gated(*args, **kwargs):
        entered.set()
        await gate.wait()
        return await real(*args, **kwargs)

    world.service.fill_cell = gated  # type: ignore[union-attr,method-assign]
    manager = manager_for(world, run_concurrency=2)
    run = await manager.start(world.workspace, RunScope(), confirmed=True)
    await entered.wait()

    await manager.stop(world.workspace, run.id)
    gate.set()
    record = await wait_run(manager, world, run.id)

    assert record.status == "stopped"
    assert (
        record.done == 2 and record.skipped == 4
    )  # the two in flight finished; four never began


async def test_no_more_cells_run_at_once_than_the_limit(world):
    await world.acme_document()
    await make_columns(world, 9)
    world.script(good())
    running, peak = 0, 0
    real = world.service.fill_cell  # type: ignore[union-attr]

    async def counted(*args, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        try:
            return await real(*args, **kwargs)
        finally:
            running -= 1

    world.service.fill_cell = counted  # type: ignore[union-attr,method-assign]
    manager = manager_for(world, run_concurrency=3)

    run = await manager.start(world.workspace, RunScope(), confirmed=True)
    record = await wait_run(manager, world, run.id)

    assert record.done == 9 and peak == 3


async def test_one_failing_cell_does_not_stop_the_run_and_is_recorded_with_a_reason(
    world,
):
    await world.acme_document()
    columns = await make_columns(world, 3)
    world.script(good())
    real = world.service.fill_cell  # type: ignore[union-attr]

    async def sometimes_broken(workspace_id, entity_id, column_id, **kwargs):
        if column_id == columns[1].id:
            raise RuntimeError("secret internal detail")
        return await real(workspace_id, entity_id, column_id, **kwargs)

    world.service.fill_cell = sometimes_broken  # type: ignore[union-attr,method-assign]
    manager = manager_for(world)

    record = await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )

    assert (record.status, record.done, record.failed) == ("completed", 3, 1)
    broken = await mrepo.get_cell_for(world.pool, world.acme.id, columns[1].id)
    assert broken and broken.status == "pending_or_error"
    assert "secret" not in (broken.status_detail or "") and "unexpectedly" in (
        broken.status_detail or ""
    )


async def test_a_cell_that_takes_too_long_is_given_up_on(world):
    await world.acme_document()
    columns = await make_columns(world, 1)

    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    world.service.fill_cell = slow  # type: ignore[union-attr,method-assign]
    manager = manager_for(world, run_cell_timeout_s=0.1)

    record = await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )

    cell = await mrepo.get_cell_for(world.pool, world.acme.id, columns[0].id)
    assert record.failed == 1 and cell and "time limit" in (cell.status_detail or "")


async def test_a_cell_pinned_after_the_plan_was_made_is_skipped_not_failed(world):
    await world.acme_document()
    await make_columns(world, 1)

    async def pinned(*args, **kwargs):
        raise mrepo.CellPinnedError("x")

    world.service.fill_cell = pinned  # type: ignore[union-attr,method-assign]
    manager = manager_for(world)

    record = await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )

    assert (record.done, record.skipped, record.failed) == (0, 1, 0)


# --- permission, one at a time, restarts -------------------------------------------------------


async def test_a_big_run_needs_a_yes_and_the_plan_comes_with_the_refusal(world):
    await make_columns(world, 3)
    manager = manager_for(world, run_confirm_above_cells=2)

    with pytest.raises(ConfirmationRequired) as needed:
        await manager.start(world.workspace, RunScope())

    assert len(needed.value.plan.cells) == 3
    assert (
        await rrepo.list_runs(world.pool, world.workspace) == []
    )  # nothing was started
    world.script(good())
    await wait_run(
        manager,
        world,
        (await manager.start(world.workspace, RunScope(), confirmed=True)).id,
    )


async def test_only_one_run_goes_at_a_time_in_a_workspace(world):
    await world.acme_document()
    await make_columns(world, 2)
    world.script(good())
    gate, entered = asyncio.Event(), asyncio.Event()
    real = world.service.fill_cell  # type: ignore[union-attr]

    async def gated(*args, **kwargs):
        entered.set()
        await gate.wait()
        return await real(*args, **kwargs)

    world.service.fill_cell = gated  # type: ignore[union-attr,method-assign]
    manager = manager_for(world)
    first = await manager.start(world.workspace, RunScope(), confirmed=True)
    await entered.wait()

    with pytest.raises(RunInProgressError):
        await manager.start(world.workspace, RunScope(), confirmed=True)

    gate.set()
    await wait_run(manager, world, first.id)


async def test_a_run_left_going_by_a_crash_is_marked_interrupted_at_start_up(world):
    stuck = await rrepo.create_run(
        world.pool, world.workspace, {}, total=5, cap_usd=D("1"), estimate_usd=None
    )

    assert await manager_for(world).recover() == 1

    record = await rrepo.get_run(world.pool, stuck.id)
    assert record and record.status == "interrupted" and record.finished_at is not None


async def test_closing_the_manager_marks_a_running_run_interrupted(world):
    await world.acme_document()
    await make_columns(world, 2)
    world.script(good())
    entered = asyncio.Event()

    async def hangs(*args, **kwargs):
        entered.set()
        await asyncio.sleep(60)

    world.service.fill_cell = hangs  # type: ignore[union-attr,method-assign]
    manager = manager_for(world)
    run = await manager.start(world.workspace, RunScope(), confirmed=True)
    await entered.wait()

    await manager.close(timeout=0.2)

    record = await rrepo.get_run(world.pool, run.id)
    assert record and record.status == "interrupted"


async def test_runs_can_be_listed_and_one_workspace_cannot_see_anothers(world):
    await make_columns(world, 1)
    manager = manager_for(world)
    run = await manager.start(world.workspace, RunScope(), confirmed=True)
    await wait_run(manager, world, run.id)
    other = await repo.create_workspace(world.pool, "Other")

    assert [r.id for r in await manager.recent(world.workspace)] == [run.id]
    assert await manager.get(other.id, run.id) is None
    assert await manager.recent(other.id) == []
    with pytest.raises(LookupError):
        await manager.stop(other.id, run.id)


# --- HTTP --------------------------------------------------------------------------------------


@pytest.fixture
async def client(world) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_pool] = lambda: world.pool
    app.dependency_overrides[get_runs] = lambda: manager_for(world)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_http_plan_then_run_then_read_the_result(client, world):
    await world.acme_document()
    await make_columns(world, 2)
    world.script(good())
    base = f"/workspaces/{world.workspace}/runs"

    plan = (await client.post(f"{base}/plan", json={})).json()
    assert (
        plan["cells"] == 2
        and plan["by_reason"] == {"missing": 2}
        and len(plan["sample"]) == 2
    )

    started = await client.post(base, json={"confirmed": True})
    assert started.status_code == 202
    run_id = started.json()["id"]
    for _ in range(300):
        body = (await client.get(f"{base}/{run_id}")).json()
        if body["status"] != "running":
            break
        await asyncio.sleep(0.03)
    assert body["status"] == "completed" and body["done"] == 2
    assert [r["id"] for r in (await client.get(base)).json()] == [run_id]


async def test_http_a_run_that_needs_a_yes_answers_409_with_the_plan(client, world):
    await make_columns(world, 2)  # unknown cost, so it needs a yes
    base = f"/workspaces/{world.workspace}/runs"

    refused = await client.post(base, json={})

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert "needs a yes" in detail["message"] and detail["plan"]["cells"] == 2
    assert (await client.get(base)).json() == []


async def test_http_bad_requests_are_plain_errors(client, world):
    await make_columns(world, 1)
    base = f"/workspaces/{world.workspace}/runs"
    missing = "00000000-0000-4000-8000-000000000000"

    assert (
        await client.post(f"{base}/plan", json={"scope": {"entity_ids": [missing]}})
    ).status_code == 422
    assert (
        await client.post(f"{base}/plan", json={"max_cost_usd": "0"})
    ).status_code == 422
    assert (
        await client.post(f"{base}/plan", json={"scope": {"entity_ids": []}})
    ).status_code == 422
    assert (
        await client.post(f"/workspaces/{missing}/runs/plan", json={})
    ).status_code == 404
    assert (await client.get(f"{base}/{missing}")).status_code == 404
    assert (await client.post(f"{base}/{missing}/stop")).status_code == 404


async def test_http_a_run_can_be_stopped(client, world):
    await world.acme_document()
    await make_columns(world, 4)
    world.script(good())
    gate, entered = asyncio.Event(), asyncio.Event()
    real = world.service.fill_cell  # type: ignore[union-attr]

    async def gated(*args, **kwargs):
        entered.set()
        await gate.wait()
        return await real(*args, **kwargs)

    world.service.fill_cell = gated  # type: ignore[union-attr,method-assign]
    manager = manager_for(world)
    app.dependency_overrides[get_runs] = lambda: manager
    base = f"/workspaces/{world.workspace}/runs"
    run_id = (await client.post(base, json={"confirmed": True})).json()["id"]
    await entered.wait()

    stopped = await client.post(f"{base}/{run_id}/stop")
    gate.set()
    for _ in range(300):
        body = (await client.get(f"{base}/{run_id}")).json()
        if body["status"] != "running":
            break
        await asyncio.sleep(0.03)

    assert stopped.status_code == 202 and body["status"] == "stopped"
