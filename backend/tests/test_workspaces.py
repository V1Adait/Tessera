"""Workspaces and entities: repository behaviour and HTTP routes. Needs the database (conftest)."""

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import asyncpg
import httpx
import pytest

from backend.app import chat_service
from backend.app.config import get_settings
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.llm import get_default_llm
from backend.app.main import app
from backend.tests.fakes import ScriptedFake, make_settings

# --- the migration ----------------------------------------------------------------------


async def test_a_default_workspace_exists_after_migrating(pool):
    workspaces = await repo.list_workspaces(pool)

    assert [w.name for w in workspaces] == ["Default"]


async def test_new_conversations_land_in_the_default_workspace_unless_told_otherwise(
    pool,
):
    default = (await repo.list_workspaces(pool))[0]
    other = await repo.create_workspace(pool, "Payments coverage")

    in_default = await repo.create_conversation(pool, "a")
    in_other = await repo.create_conversation(pool, "b", workspace_id=other.id)

    first = await repo.get_conversation(pool, in_default)
    second = await repo.get_conversation(pool, in_other)
    assert first is not None and first.workspace_id == default.id
    assert second is not None and second.workspace_id == other.id


async def test_a_conversation_in_an_unknown_workspace_is_refused(pool):
    with pytest.raises(repo.WorkspaceNotFoundError):
        await repo.create_conversation(pool, "x", workspace_id=uuid4())


# --- entities ---------------------------------------------------------------------------


async def test_entities_keep_their_insertion_order(pool):
    ws = await repo.create_workspace(pool, "w")
    for name in ["Visa", "Mastercard", "PayPal"]:
        await repo.create_entity(pool, ws.id, name)

    names = [e.name for e in await repo.list_entities(pool, ws.id)]

    assert names == ["Visa", "Mastercard", "PayPal"]


async def test_entity_names_are_trimmed_and_unique_ignoring_case_and_spaces(pool):
    ws = await repo.create_workspace(pool, "w")
    first = await repo.create_entity(pool, ws.id, "  Visa  ")
    assert first.name == "Visa"

    for duplicate in ["visa", "VISA", " Visa "]:
        with pytest.raises(repo.DuplicateNameError):
            await repo.create_entity(pool, ws.id, duplicate)


async def test_the_same_entity_name_is_allowed_in_another_workspace(pool):
    one = await repo.create_workspace(pool, "one")
    two = await repo.create_workspace(pool, "two")

    await repo.create_entity(pool, one.id, "Visa")
    await repo.create_entity(pool, two.id, "Visa")  # no error

    assert len(await repo.list_entities(pool, one.id)) == 1
    assert len(await repo.list_entities(pool, two.id)) == 1


async def test_an_entity_in_an_unknown_workspace_is_refused(pool):
    with pytest.raises(repo.WorkspaceNotFoundError):
        await repo.create_entity(pool, uuid4(), "Visa")


async def test_concurrent_entity_creation_never_shares_a_position_or_a_name(pool):
    ws = await repo.create_workspace(pool, "w")

    results = await asyncio.gather(
        *(repo.create_entity(pool, ws.id, f"Company {i}") for i in range(20)),
        *(repo.create_entity(pool, ws.id, "Same Name") for _ in range(5)),
        return_exceptions=True,
    )

    created = [r for r in results if not isinstance(r, Exception)]
    duplicates = [r for r in results if isinstance(r, repo.DuplicateNameError)]
    assert len(created) == 21 and len(duplicates) == 4  # exactly one "Same Name" won
    assert len({e.position for e in created}) == 21  # no shared row position


async def test_the_database_itself_rejects_blank_and_oversized_names(pool):
    ws = await repo.create_workspace(pool, "w")
    async with pool.acquire() as conn:
        for bad in ["", "   ", "x" * 201]:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    "INSERT INTO entities (workspace_id, name) VALUES ($1, $2)",
                    ws.id,
                    bad,
                )


async def test_deleting_a_workspace_removes_its_entities_and_conversations(pool):
    ws = await repo.create_workspace(pool, "doomed")
    await repo.create_entity(pool, ws.id, "Visa")
    conv = await repo.create_conversation(pool, "c", workspace_id=ws.id)
    await repo.add_message(pool, conv, "user", "hi")

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM workspaces WHERE id = $1", ws.id)
        counts = await conn.fetchrow(
            "SELECT (SELECT count(*) FROM entities WHERE workspace_id = $1) AS e, "
            "(SELECT count(*) FROM conversations WHERE workspace_id = $1) AS c, "
            "(SELECT count(*) FROM messages WHERE conversation_id = $2) AS m",
            ws.id,
            conv,
        )
    assert counts is not None
    assert (counts["e"], counts["c"], counts["m"]) == (0, 0, 0)


# --- chat uses the workspace ------------------------------------------------------------


async def test_begin_turn_puts_a_new_conversation_in_the_requested_workspace(pool):
    ws = await repo.create_workspace(pool, "w")

    turn = await chat_service.begin_turn(pool, None, "hi", uuid4(), workspace_id=ws.id)

    conversation = await repo.get_conversation(pool, turn.conversation_id)
    assert conversation is not None and conversation.workspace_id == ws.id


async def test_begin_turn_with_an_unknown_workspace_creates_nothing(pool):
    with pytest.raises(chat_service.WorkspaceNotFoundError):
        await chat_service.begin_turn(pool, None, "hi", uuid4(), workspace_id=uuid4())

    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM conversations") == 0
        assert await conn.fetchval("SELECT count(*) FROM messages") == 0


# --- HTTP -------------------------------------------------------------------------------


@pytest.fixture
async def client(pool) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_settings] = make_settings
    app.dependency_overrides[get_default_llm] = lambda: ScriptedFake()
    app.dependency_overrides[get_pool] = lambda: pool
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_http_workspace_and_entity_flow(client):
    created = await client.post("/workspaces", json={"name": "  Fintech  "})
    assert created.status_code == 201
    workspace = created.json()
    assert workspace["name"] == "Fintech"  # trimmed

    listed = (await client.get("/workspaces")).json()
    assert {w["name"] for w in listed} == {"Default", "Fintech"}

    first = await client.post(
        f"/workspaces/{workspace['id']}/entities", json={"name": "Visa"}
    )
    assert first.status_code == 201
    await client.post(f"/workspaces/{workspace['id']}/entities", json={"name": "Adyen"})
    entities = (await client.get(f"/workspaces/{workspace['id']}/entities")).json()
    assert [e["name"] for e in entities] == ["Visa", "Adyen"]


async def test_http_duplicate_entity_is_409(client):
    ws = (await client.post("/workspaces", json={"name": "w"})).json()
    url = f"/workspaces/{ws['id']}/entities"
    assert (await client.post(url, json={"name": "Visa"})).status_code == 201

    duplicate = await client.post(url, json={"name": " visa "})

    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "an entity with this name already exists"}


async def test_http_unknown_workspace_is_404_everywhere(client):
    missing = uuid4()
    assert (await client.get(f"/workspaces/{missing}/entities")).status_code == 404
    assert (
        await client.post(f"/workspaces/{missing}/entities", json={"name": "Visa"})
    ).status_code == 404
    assert (
        await client.post("/chat", json={"message": "hi", "workspace_id": str(missing)})
    ).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 201},
        {"name": "bad\x00name"},
        {"name": "line\nbreak"},
        {"name": 42},
    ],
)
async def test_http_bad_entity_names_are_422(client, body):
    ws = (await client.post("/workspaces", json={"name": "w"})).json()

    r = await client.post(f"/workspaces/{ws['id']}/entities", json=body)

    assert r.status_code == 422


@pytest.mark.parametrize("body", [{}, {"name": ""}, {"name": "x" * 121}])
async def test_http_bad_workspace_names_are_422(client, body):
    assert (await client.post("/workspaces", json=body)).status_code == 422


async def test_http_sql_looking_names_are_stored_as_plain_text(client, pool):
    ws = (await client.post("/workspaces", json={"name": "w"})).json()
    nasty = "x'); DROP TABLE entities; --"

    r = await client.post(f"/workspaces/{ws['id']}/entities", json={"name": nasty})

    assert r.status_code == 201 and r.json()["name"] == nasty
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM entities") == 1


async def test_chat_rejects_conversation_and_workspace_together(client):
    body = {
        "message": "hi",
        "conversation_id": str(uuid4()),
        "workspace_id": str(uuid4()),
    }
    assert (await client.post("/chat", json=body)).status_code == 422


async def test_chat_in_a_chosen_workspace_and_the_view_reports_it(client):
    ws = (await client.post("/workspaces", json={"name": "w"})).json()

    r = await client.post("/chat", json={"message": "hi", "workspace_id": ws["id"]})
    view = await client.get(f"/conversations/{r.headers['x-conversation-id']}")

    assert view.json()["workspace_id"] == ws["id"]
