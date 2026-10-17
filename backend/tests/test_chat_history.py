"""The chat history: listing a workspace's conversations, and deleting one. Needs the database."""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest

from backend.app.db import agent_repository as arepo
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import TraceRecord
from backend.app.main import app


@pytest.fixture
async def client(pool) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_pool] = lambda: pool
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def chat(pool, workspace, title, *messages):
    conversation = await repo.create_conversation(pool, title, workspace)
    for role, text in messages:
        await repo.add_message(pool, conversation, role, text)
    return conversation


# --- listing ----------------------------------------------------------------------------------------------------------


async def test_the_history_lists_a_workspaces_conversations_newest_first_with_message_counts(
    pool,
):
    workspace = await repo.default_workspace_id(pool)
    first = await chat(
        pool, workspace, "How leveraged is Palantir?", ("user", "q"), ("assistant", "a")
    )
    second = await chat(pool, workspace, "Revenue of Acme", ("user", "q"))

    listed = await repo.list_conversations(pool, workspace)

    assert [(c.id, c.title, c.message_count) for c in listed] == [
        (second, "Revenue of Acme", 1),
        (first, "How leveraged is Palantir?", 2),
    ]


async def test_a_conversation_moves_to_the_top_when_it_is_used_again(pool):
    workspace = await repo.default_workspace_id(pool)
    old = await chat(pool, workspace, "old", ("user", "q"))
    await chat(pool, workspace, "newer", ("user", "q"))

    await repo.add_message(pool, old, "assistant", "a late answer")

    assert [c.title for c in await repo.list_conversations(pool, workspace)] == [
        "old",
        "newer",
    ]


async def test_a_conversation_with_no_messages_is_not_history(pool):
    workspace = await repo.default_workspace_id(pool)
    await repo.create_conversation(pool, "empty", workspace)
    kept = await chat(pool, workspace, "real", ("user", "q"))

    assert [c.id for c in await repo.list_conversations(pool, workspace)] == [kept]


async def test_one_workspace_never_sees_another_workspaces_history(pool):
    mine = await repo.default_workspace_id(pool)
    other = await pool.fetchval(
        "insert into workspaces (name) values ('Other') returning id"
    )
    await chat(pool, mine, "mine", ("user", "q"))
    await chat(pool, other, "theirs", ("user", "q"))

    assert [c.title for c in await repo.list_conversations(pool, mine)] == ["mine"]
    assert [c.title for c in await repo.list_conversations(pool, other)] == ["theirs"]


async def test_the_list_is_limited(pool):
    workspace = await repo.default_workspace_id(pool)
    for i in range(5):
        await chat(pool, workspace, f"c{i}", ("user", "q"))

    assert len(await repo.list_conversations(pool, workspace, limit=3)) == 3


async def test_the_route_returns_the_history_and_404s_for_an_unknown_workspace(
    client, pool
):
    workspace = await repo.default_workspace_id(pool)
    await chat(pool, workspace, "hello", ("user", "q"), ("assistant", "a"))

    ok = await client.get(f"/workspaces/{workspace}/conversations")
    missing = await client.get(f"/workspaces/{uuid4()}/conversations")

    assert ok.status_code == 200
    body = ok.json()
    assert [(c["title"], c["message_count"]) for c in body] == [("hello", 2)]
    assert set(body[0]) == {"id", "title", "updated_at", "message_count"}
    assert missing.status_code == 404


# --- deleting ---------------------------------------------------------------------------------------------------------------


async def test_deleting_removes_the_conversation_and_its_messages(client, pool):
    workspace = await repo.default_workspace_id(pool)
    gone = await chat(pool, workspace, "gone", ("user", "q"), ("assistant", "a"))
    kept = await chat(pool, workspace, "kept", ("user", "q"))

    r = await client.delete(f"/conversations/{gone}")

    assert r.status_code == 204
    assert (
        await pool.fetchval(
            "select count(*) from messages where conversation_id = $1", gone
        )
        == 0
    )
    assert [c.id for c in await repo.list_conversations(pool, workspace)] == [kept]
    assert (await client.get(f"/conversations/{gone}")).status_code == 404


async def test_deleting_an_unknown_conversation_is_a_404(client):
    assert (await client.delete(f"/conversations/{uuid4()}")).status_code == 404


async def test_deleting_takes_pending_confirmations_with_it(client, pool):
    workspace = await repo.default_workspace_id(pool)
    conversation = await chat(pool, workspace, "c", ("user", "q"))
    await arepo.create_action(
        pool,
        conversation_id=conversation,
        workspace_id=workspace,
        request_id=uuid4(),
        tool="remove_row",
        args={},
        risk="destructive",
        summary="s",
    )

    await client.delete(f"/conversations/{conversation}")

    assert await pool.fetchval("select count(*) from agent_actions") == 0


async def test_deleting_keeps_the_cost_record(client, pool):
    """Traces are the record of what was spent: deleting the chat must not erase the spend."""
    workspace = await repo.default_workspace_id(pool)
    conversation = await chat(pool, workspace, "c", ("user", "q"))
    await repo.record_trace(
        pool,
        TraceRecord(
            request_id=uuid4(),
            kind="chat",
            status="ok",
            conversation_id=conversation,
            tokens_in=10,
            tokens_out=5,
        ),
    )

    await client.delete(f"/conversations/{conversation}")

    row = await pool.fetchrow("select conversation_id, tokens_in from traces")
    assert row is not None and row["tokens_in"] == 10 and row["conversation_id"] is None
