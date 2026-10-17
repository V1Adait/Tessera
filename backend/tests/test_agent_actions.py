"""Confirming and cancelling what the agent was not allowed to do alone, and the HTTP routes
around the agent. A real database; the model is scripted. The point of these tests: nothing
dangerous happens without a yes, and a yes happens exactly once.
"""

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import pytest

from backend.app.agent.actions import (
    ActionDesk,
    ActionExpiredError,
    ActionNotFoundError,
    ActionNotPendingError,
)
from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.tools import Services
from backend.app.api_agent import get_agent
from backend.app.config import get_settings
from backend.app.db import agent_repository as arepo
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.documents_service import DocumentService
from backend.app.llm import get_default_llm
from backend.app.main import app
from backend.app.matrix.runs import RunManager
from backend.app.models import ConfirmationEvent
from backend.tests.fakes import ScriptedFake, ToolFake, make_settings
from backend.tests.test_agent_loop import Rig, of
from backend.tests.test_api import parse_sse
from backend.tests.test_matrix_flow import World, good
from backend.tests.test_matrix_runs import wait_run


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


async def rig_for(world: World, pool, **settings) -> Rig:
    return Rig(
        world, await repo.create_conversation(pool, "t", world.workspace), **settings
    )


def desk_for(rig: Rig) -> ActionDesk:
    return ActionDesk(rig.ctx.pool, rig.settings, rig.ctx.services, rig.deps.registry)


async def ask_to_remove_acme(rig: Rig):
    rig.script([("remove_row", {"row": "Acme"})], "Waiting for your confirmation.")
    events = await rig.run()
    return of(events, ConfirmationEvent)[0]


# --- confirm ------------------------------------------------------------------------------------------


async def test_confirming_runs_the_stored_action_and_tells_the_conversation(
    world, pool
):
    rig = await rig_for(world, pool)
    ask = await ask_to_remove_acme(rig)

    outcome = await desk_for(rig).confirm(ask.action_id)

    assert (
        outcome.ok
        and outcome.changed_matrix
        and "Removed row 'Acme'" in outcome.message
    )
    assert await repo.list_entities(pool, world.workspace) == []
    assert outcome.action.status == "confirmed"
    said = await repo.list_messages(pool, rig.ctx.conversation_id)
    assert said[-1].role == "assistant" and "Removed row 'Acme'" in said[-1].content


async def test_two_confirmations_at_once_run_the_action_exactly_once(world, pool):
    rig = await rig_for(world, pool)
    ask = await ask_to_remove_acme(rig)
    desk = desk_for(rig)

    results = await asyncio.gather(
        desk.confirm(ask.action_id), desk.confirm(ask.action_id), return_exceptions=True
    )

    wins = [r for r in results if not isinstance(r, Exception)]
    losses = [r for r in results if isinstance(r, ActionNotPendingError)]
    assert len(wins) == 1 and len(losses) == 1


async def test_a_confirmed_action_cannot_be_confirmed_again(world, pool):
    rig = await rig_for(world, pool)
    ask = await ask_to_remove_acme(rig)
    desk = desk_for(rig)
    await desk.confirm(ask.action_id)

    with pytest.raises(ActionNotPendingError):
        await desk.confirm(ask.action_id)


async def test_an_unknown_action_is_not_found(world, pool):
    rig = await rig_for(world, pool)

    with pytest.raises(ActionNotFoundError):
        await desk_for(rig).confirm(uuid4())


async def test_an_old_action_is_refused_and_dropped(world, pool):
    rig = await rig_for(world, pool, agent_action_ttl_minutes=0)
    ask = await ask_to_remove_acme(rig)
    await asyncio.sleep(0.01)

    with pytest.raises(ActionExpiredError):
        await desk_for(rig).confirm(ask.action_id)

    assert len(await repo.list_entities(pool, world.workspace)) == 1
    stored = await arepo.get_action(pool, ask.action_id)
    assert stored is not None and stored.status == "cancelled"


async def test_if_the_row_vanished_meanwhile_the_confirmation_fails_cleanly(
    world, pool
):
    rig = await rig_for(world, pool)
    ask = await ask_to_remove_acme(rig)
    await repo.delete_entity(pool, world.workspace, world.acme.id)

    outcome = await desk_for(rig).confirm(ask.action_id)

    assert not outcome.ok and "no row called 'Acme'" in outcome.message
    stored = await arepo.get_action(pool, ask.action_id)
    assert stored is not None and stored.status == "failed"
    said = await repo.list_messages(pool, rig.ctx.conversation_id)
    assert "Could not do it" in said[-1].content


async def test_confirming_a_big_fill_starts_the_run(world, pool):
    rig = await rig_for(world, pool, run_confirm_above_cells=0)
    await world.acme_document()
    await world.column()
    world.script(good())
    rig.script([("fill_cells", {})], "Waiting.")
    ask = of(await rig.run(), ConfirmationEvent)[0]
    assert await pool.fetchval("select count(*) from matrix_runs") == 0

    outcome = await desk_for(rig).confirm(ask.action_id)

    assert outcome.ok and "Started filling 1 cells" in outcome.message
    manager = rig.ctx.services.runs
    run = (await manager.recent(world.workspace))[0]  # type: ignore[union-attr]
    finished = await wait_run(manager, world, run.id)  # type: ignore[arg-type]
    assert finished.status == "completed" and finished.done == 1


# --- cancel --------------------------------------------------------------------------------------------


async def test_cancelling_changes_nothing_and_says_so(world, pool):
    rig = await rig_for(world, pool)
    ask = await ask_to_remove_acme(rig)

    outcome = await desk_for(rig).cancel(ask.action_id)

    assert outcome.ok and outcome.action.status == "cancelled"
    assert len(await repo.list_entities(pool, world.workspace)) == 1
    said = await repo.list_messages(pool, rig.ctx.conversation_id)
    assert "Cancelled" in said[-1].content


async def test_a_cancelled_action_cannot_be_confirmed(world, pool):
    rig = await rig_for(world, pool)
    ask = await ask_to_remove_acme(rig)
    desk = desk_for(rig)
    await desk.cancel(ask.action_id)

    with pytest.raises(ActionNotPendingError):
        await desk.confirm(ask.action_id)
    assert len(await repo.list_entities(pool, world.workspace)) == 1


async def test_pending_lists_only_what_is_still_waiting(world, pool):
    rig = await rig_for(world, pool)
    first = await ask_to_remove_acme(rig)
    rig.script([("remove_column", {"column": "Nope"})], "x")
    desk = desk_for(rig)
    assert [a.id for a in await desk.pending(rig.ctx.conversation_id)] == [
        first.action_id
    ]

    await desk.cancel(first.action_id)

    assert await desk.pending(rig.ctx.conversation_id) == []


# --- over HTTP -------------------------------------------------------------------------------------------


@pytest.fixture
async def http(
    world, pool, settings
) -> AsyncIterator[tuple[httpx.AsyncClient, ToolFake]]:
    llm = ToolFake(steps=["hi"])
    runtime = AgentRuntime(
        pool,
        settings,
        Services(
            matrix=world.service,
            runs=RunManager(pool, settings, world.service),  # type: ignore[arg-type]
            documents=world.documents,
        ),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_default_llm] = lambda: llm
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_agent] = lambda: runtime
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, llm
    app.dependency_overrides.clear()


async def test_chat_runs_the_agent_in_the_workspace_the_client_names(http, world, pool):
    client, llm = http
    llm.steps[:] = [[("add_row", {"name": "Globex"})], "Added Globex."]

    r = await client.post(
        "/chat", json={"message": "add Globex", "workspace_id": str(world.workspace)}
    )

    events = parse_sse(r.text)
    kinds = [e["type"] for e in events]
    assert "tool" in kinds and kinds[-1] == "done"
    tool_done = next(e for e in events if e["type"] == "tool" and e["status"] == "ok")
    assert tool_done["changed_matrix"] is True and tool_done["name"] == "add_row"
    assert "Globex" in [e.name for e in await repo.list_entities(pool, world.workspace)]
    # one trace per model call, none for the turn as a whole (that would count the cost twice)
    assert await pool.fetchval("select count(*) from traces where kind = 'agent'") == 2
    assert await pool.fetchval("select count(*) from traces where kind = 'chat'") == 0
    # only the person's message and the final answer are kept as the conversation
    said = await repo.list_messages(pool, UUID(r.headers["x-conversation-id"]))
    assert [(m.role, m.content) for m in said] == [
        ("user", "add Globex"),
        ("assistant", "Added Globex."),
    ]


async def test_the_confirmation_arrives_in_the_stream_and_is_resolved_over_http(
    http, world, pool
):
    client, llm = http
    llm.steps[:] = [[("remove_row", {"row": "Acme"})], "Please confirm."]
    r = await client.post("/chat", json={"message": "remove Acme"})
    conversation = r.headers["x-conversation-id"]
    ask = next(e for e in parse_sse(r.text) if e["type"] == "confirmation")
    assert (
        ask["risk"] == "destructive"
        and len(await repo.list_entities(pool, world.workspace)) == 1
    )

    pending = (await client.get(f"/conversations/{conversation}/actions")).json()
    assert [a["id"] for a in pending] == [ask["action_id"]]

    done = await client.post(f"/actions/{ask['action_id']}/confirm")
    assert done.status_code == 200 and done.json()["ok"] is True
    assert await repo.list_entities(pool, world.workspace) == []

    again = await client.post(f"/actions/{ask['action_id']}/confirm")
    assert again.status_code == 409
    assert (await client.post(f"/actions/{uuid4()}/cancel")).status_code == 404
    assert (await client.get(f"/conversations/{conversation}/actions")).json() == []


async def test_cancelling_over_http_leaves_the_row(http, world, pool):
    client, llm = http
    llm.steps[:] = [[("remove_row", {"row": "Acme"})], "Please confirm."]
    r = await client.post("/chat", json={"message": "remove Acme"})
    ask = next(e for e in parse_sse(r.text) if e["type"] == "confirmation")

    done = await client.post(f"/actions/{ask['action_id']}/cancel")

    assert done.status_code == 200 and done.json()["action"]["status"] == "cancelled"
    assert len(await repo.list_entities(pool, world.workspace)) == 1


async def test_a_conversation_over_its_spending_cap_is_refused_before_any_model_call(
    http, world, pool
):
    client, llm = http
    llm.steps[:] = ["hi"]
    first = await client.post("/chat", json={"message": "hello"})
    conversation = first.headers["x-conversation-id"]
    await pool.execute(
        "update traces set cost_usd = 5, cost_source = 'provider' where kind = 'agent'"
    )
    calls = len(llm.seen)

    r = await client.post(
        "/chat", json={"message": "more", "conversation_id": conversation}
    )

    events = parse_sse(r.text)
    assert events[-1]["type"] == "error" and events[-1]["code"] == "budget"
    assert len(llm.seen) == calls


async def test_a_model_without_tool_support_gets_a_clear_error(world, pool, settings):
    runtime = AgentRuntime(pool, settings, Services(matrix=world.service))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_default_llm] = lambda: (
        ScriptedFake()
    )  # cannot bind tools
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_agent] = lambda: runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/chat", json={"message": "hi"})
    finally:
        app.dependency_overrides.clear()

    last = parse_sse(r.text)[-1]
    assert last["type"] == "error" and "cannot use tools" in last["message"]
