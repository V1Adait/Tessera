"""Removing a column: what goes with it, and when it is refused. Real database, scripted model."""

import json
from collections.abc import AsyncIterator
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest

from backend.app.api_matrix import get_matrix
from backend.app.config import get_settings
from backend.app.db import matrix_repository as mrepo
from backend.app.db.pool import get_pool
from backend.app.documents_service import DocumentService
from backend.app.finance.checks import CheckSpec
from backend.app.finance.derive import Derivation
from backend.app.main import app
from backend.app.matrix.models import ColumnDraft
from backend.app.matrix.service import ColumnBusyError
from backend.tests.fakes import make_settings
from backend.tests.test_matrix_flow import World

REVENUE = "Revenue for the quarter ended June 30, 2026 was $2,000 million."
COST = "Cost of revenue for the quarter ended June 30, 2026 was $800 million."
GROSS = "Gross profit for the quarter ended June 30, 2026 was $1,200 million."


def says(answer: str, quote: str) -> str:
    return json.dumps(
        {
            "status": "answered",
            "answer": answer,
            "period": "Q2 2026",
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
    await w.acme_document(REVENUE, COST, GROSS)
    return w


async def money(world: World, name: str):
    return await world.column(
        name=name, prompt=f"What was {name.lower()}?", type="currency"
    )


async def fill(world: World, column, reply: str):
    world.script(reply)
    return await world.service.fill_cell(world.workspace, world.acme.id, column.id)  # type: ignore[union-attr]


async def columns_of(world: World):
    return [c.name for c in await mrepo.list_columns(world.pool, world.workspace)]


# --- the service -------------------------------------------------------------------------------------------------


async def test_removing_a_column_removes_its_cells_and_leaves_the_others(world):
    revenue, cost = await money(world, "Revenue"), await money(world, "Cost of revenue")
    await fill(world, revenue, says("$2,000 million", REVENUE))
    await fill(world, cost, says("$800 million", COST))

    await world.service.delete_column(world.workspace, cost.id)  # type: ignore[union-attr]

    assert await columns_of(world) == ["Revenue"]
    left = await mrepo.list_cells(world.pool, world.workspace)
    assert [c.column_id for c in left] == [
        revenue.id
    ]  # the other column's cell is untouched


async def test_an_unknown_column_is_not_found(world):
    with pytest.raises(mrepo.ColumnNotFoundError):
        await world.service.delete_column(world.workspace, uuid4())  # type: ignore[union-attr]


async def test_a_column_of_another_workspace_is_not_found_and_not_removed(world, pool):
    other = await pool.fetchval(
        "insert into workspaces (name) values ('Other') returning id"
    )
    column = await money(world, "Revenue")

    with pytest.raises(mrepo.ColumnNotFoundError):
        await world.service.delete_column(other, column.id)  # type: ignore[union-attr]

    assert await columns_of(world) == ["Revenue"]


async def test_nothing_is_removed_while_a_fill_is_going(world, pool):
    column = await money(world, "Revenue")
    await pool.execute(
        "insert into matrix_runs (workspace_id, status, scope, total, cap_usd) "
        "values ($1, 'running', '{}', 1, 1)",
        world.workspace,
    )

    with pytest.raises(ColumnBusyError, match="A fill is going"):
        await world.service.delete_column(world.workspace, column.id)  # type: ignore[union-attr]

    assert await columns_of(world) == ["Revenue"]


async def test_a_column_that_a_computed_column_is_built_on_cannot_be_removed(world):
    revenue, cost = await money(world, "Revenue"), await money(world, "Cost of revenue")
    await world.service.create_column(  # type: ignore[union-attr]
        world.workspace,
        ColumnDraft(
            name="Cost ratio",
            prompt="Cost over revenue, computed.",
            type="number",
            depends_on=[cost.id, revenue.id],
            derivation=Derivation(op="ratio", **{"as": "percent"}),  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(ColumnBusyError, match=r"Cost ratio is computed from 'Revenue'"):
        await world.service.delete_column(world.workspace, revenue.id)  # type: ignore[union-attr]

    assert len(await columns_of(world)) == 3


async def test_removing_the_computed_column_itself_is_fine_and_leaves_its_inputs(world):
    revenue, cost = await money(world, "Revenue"), await money(world, "Cost of revenue")
    ratio = await world.service.create_column(  # type: ignore[union-attr]
        world.workspace,
        ColumnDraft(
            name="Cost ratio",
            prompt="computed",
            type="number",
            depends_on=[cost.id, revenue.id],
            derivation=Derivation(op="ratio", **{"as": "percent"}),  # type: ignore[arg-type]
        ),
    )

    await world.service.delete_column(world.workspace, ratio.id)  # type: ignore[union-attr]

    assert await columns_of(world) == ["Revenue", "Cost of revenue"]


async def test_a_tie_out_that_named_the_removed_column_disappears_from_the_cell(world):
    world.service._checks = [  # type: ignore[union-attr]
        CheckSpec(
            id="gross_ties",
            name="Gross profit = revenue - cost",
            kind="difference_identity",
            a="Revenue",
            b="Cost of revenue",
            result="Gross profit",
            tolerance=Decimal("0.005"),
        )
    ]
    revenue, cost, gross = (
        await money(world, "Revenue"),
        await money(world, "Cost of revenue"),
        await money(world, "Gross profit"),
    )
    await fill(world, revenue, says("$2,000 million", REVENUE))
    await fill(world, cost, says("$800 million", COST))
    await fill(world, gross, says("$1,200 million", GROSS))
    before = await mrepo.get_cell_for(world.pool, world.acme.id, gross.id)
    assert before is not None and [c["status"] for c in before.checks] == ["ties"]

    await world.service.delete_column(world.workspace, cost.id)  # type: ignore[union-attr]

    after = await mrepo.get_cell_for(world.pool, world.acme.id, gross.id)
    assert (
        after is not None and after.checks == []
    )  # the check no longer applies; no stale tick


# --- the route -------------------------------------------------------------------------------------------------------


@pytest.fixture
async def client(world) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_settings] = lambda: world.settings
    app.dependency_overrides[get_pool] = lambda: world.pool
    app.dependency_overrides[get_matrix] = lambda: world.service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def test_the_route_removes_a_column_and_answers_204(client, world):
    column = await money(world, "Revenue")

    r = await client.delete(f"/workspaces/{world.workspace}/columns/{column.id}")

    assert r.status_code == 204 and r.content == b""
    assert await columns_of(world) == []


async def test_the_route_404s_for_an_unknown_column(client, world):
    r = await client.delete(f"/workspaces/{world.workspace}/columns/{uuid4()}")

    assert r.status_code == 404


async def test_the_route_409s_with_the_reason_when_it_cannot_remove(
    client, world, pool
):
    column = await money(world, "Revenue")
    await pool.execute(
        "insert into matrix_runs (workspace_id, status, scope, total, cap_usd) "
        "values ($1, 'running', '{}', 1, 1)",
        world.workspace,
    )

    r = await client.delete(f"/workspaces/{world.workspace}/columns/{column.id}")

    assert r.status_code == 409 and "A fill is going" in r.json()["detail"]
    assert await columns_of(world) == ["Revenue"]
