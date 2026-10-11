"""Tests for chat_service: what gets stored around a chat turn. Needs the database (see conftest)."""

import asyncio
import contextlib
import logging
from uuid import uuid4

import pytest

from backend.app import chat_service
from backend.app.db import repository as repo
from backend.app.models import Delta, Done, ErrorEvent
from backend.app.prompts import PROMPT_VERSION
from backend.tests.fakes import BoomFake, ScriptedFake, SlowFake, make_settings


async def run_turn(pool, llm, text, conversation_id=None, settings=None):
    """begin_turn + stream_turn; returns (turn, events)."""
    turn = await chat_service.begin_turn(pool, conversation_id, text, uuid4())
    events = [
        e
        async for e in chat_service.stream_turn(
            pool, settings or make_settings(), llm, turn
        )
    ]
    return turn, events


async def wait_for(condition, timeout=3.0):
    """Poll an async condition: recording after a disconnect happens in the background."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await condition():
            return True
        await asyncio.sleep(0.02)
    return False


# --- the normal path --------------------------------------------------------------------


async def test_a_turn_stores_both_messages_and_a_trace(pool):
    turn, events = await run_turn(pool, ScriptedFake(), "What is revenue?")

    assert isinstance(events[-1], Done)
    user, assistant = await repo.list_messages(pool, turn.conversation_id)
    assert (user.role, user.content, user.status) == (
        "user",
        "What is revenue?",
        "complete",
    )
    assert (assistant.role, assistant.content, assistant.status) == (
        "assistant",
        "hello there",
        "complete",
    )
    assert assistant.tokens == 2
    assert (
        user.request_id == assistant.request_id == turn.request_id
    )  # all linked to one trace

    (trace,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert trace.request_id == turn.request_id
    assert (trace.kind, trace.status, trace.error_code) == ("chat", "ok", None)
    assert (trace.tokens_in, trace.tokens_out) == (5, 2)
    assert (trace.provider, trace.model) == ("anthropic", "m")
    assert trace.prompt_version == PROMPT_VERSION
    assert trace.latency_ms is not None and trace.first_token_ms is not None


async def test_new_conversation_gets_a_short_title(pool):
    turn, _ = await run_turn(pool, ScriptedFake(), "x" * 200)

    conversation = await repo.get_conversation(pool, turn.conversation_id)
    assert conversation is not None and conversation.title == "x" * 60


async def test_second_turn_sends_the_stored_history_to_the_model(pool):
    llm = ScriptedFake()
    first, _ = await run_turn(pool, llm, "first question")

    await run_turn(pool, llm, "second question", first.conversation_id)

    prompt = llm.seen[-1]
    assert [m.type for m in prompt] == ["system", "human", "ai", "human"]
    assert [m.content for m in prompt[1:]] == [
        "first question",
        "hello there",
        "second question",
    ]


async def test_unknown_conversation_is_rejected_and_leaves_nothing_behind(pool):
    with pytest.raises(chat_service.ConversationNotFoundError):
        await chat_service.begin_turn(pool, uuid4(), "hi", uuid4())

    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM messages") == 0


# --- the summary pointer is honoured ----------------------------------------------------


async def test_prompt_uses_the_summary_and_only_messages_after_its_pointer(pool):
    conv = await repo.create_conversation(pool)
    # two old turns, covered by the summary; the pointer sits on a turn boundary, as it does
    # when plan_summary chooses it
    await repo.add_message(pool, conv, "user", "old question 1")
    await repo.add_message(pool, conv, "assistant", "old answer 1")
    await repo.add_message(pool, conv, "user", "old question 2")
    boundary = await repo.add_message(pool, conv, "assistant", "old answer 2")
    await repo.add_summary(pool, conv, "they discussed Visa", boundary.id, "sum-v1")
    await repo.add_message(pool, conv, "user", "kept question")
    await repo.add_message(pool, conv, "assistant", "kept answer")
    llm = ScriptedFake()

    await run_turn(pool, llm, "new question", conv)

    prompt = llm.seen[-1]
    assert "they discussed Visa" in str(prompt[0].content)
    assert [m.content for m in prompt[1:]] == [
        "kept question",
        "kept answer",
        "new question",
    ]
    assert len(await repo.list_messages(pool, conv)) == 8  # nothing was deleted


# --- refusals and failures --------------------------------------------------------------


async def test_a_too_large_message_is_stored_but_traced_as_rejected(pool):
    settings = make_settings(max_context_tokens=5000)

    turn, events = await run_turn(pool, ScriptedFake(), "x" * 60_000, settings=settings)

    assert [type(e) for e in events] == [ErrorEvent]
    messages = await repo.list_messages(pool, turn.conversation_id)
    assert [m.role for m in messages] == [
        "user"
    ]  # no assistant message: the model was not called
    (trace,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert (trace.status, trace.error_code) == ("rejected", "context_too_large")
    assert trace.tokens_in is None


async def test_a_provider_failure_is_recorded_and_excluded_from_later_prompts(pool):
    turn, events = await run_turn(pool, BoomFake(), "hello")

    assert isinstance(events[-1], ErrorEvent)
    _, assistant = await repo.list_messages(pool, turn.conversation_id)
    assert (assistant.status, assistant.content) == ("error", "partial ")
    (trace,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert (trace.status, trace.error_code) == ("error", "unknown")

    llm = ScriptedFake()
    await run_turn(pool, llm, "try again", turn.conversation_id)
    assert [m.content for m in llm.seen[-1][1:]] == [
        "hello",
        "try again",
    ]  # the failed reply is gone


# --- the client goes away ---------------------------------------------------------------


async def test_closing_the_stream_early_keeps_the_partial_answer(pool):
    turn = await chat_service.begin_turn(pool, None, "count for me", uuid4())
    stream = chat_service.stream_turn(pool, make_settings(), SlowFake(), turn)

    async for event in stream:
        if isinstance(event, Delta):
            break  # the client stops reading after the first word
    await stream.aclose()

    _, assistant = await repo.list_messages(pool, turn.conversation_id)
    assert assistant.status == "stopped"
    assert assistant.content.startswith("one")
    (trace,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert (trace.status, trace.tokens_out) == ("cancelled", None)


async def test_a_cancelled_request_still_records_in_the_background(pool):
    """The real disconnect case: the request task is cancelled mid-await."""
    turn = await chat_service.begin_turn(pool, None, "count for me", uuid4())
    seen: list[object] = []

    async def consume():
        async for event in chat_service.stream_turn(
            pool, make_settings(), SlowFake(), turn
        ):
            seen.append(event)

    task = asyncio.create_task(consume())
    assert await wait_for(lambda: _non_empty(seen))
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    async def trace_exists():
        return (
            len(await repo.list_traces(pool, conversation_id=turn.conversation_id)) == 1
        )

    assert await wait_for(trace_exists)
    (trace,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert trace.status == "cancelled"
    messages = await repo.list_messages(pool, turn.conversation_id)
    assert messages[-1].role == "assistant" and messages[-1].status == "stopped"


async def _non_empty(items: list[object]) -> bool:
    return bool(items)


# --- recording must never break the chat ------------------------------------------------


async def test_a_recording_failure_does_not_break_the_stream(pool, monkeypatch, caplog):
    async def explode(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(repo, "record_trace", explode)

    with caplog.at_level(logging.ERROR):
        turn, events = await run_turn(pool, ScriptedFake(), "hi")

    assert isinstance(events[-1], Done)  # the user still got the full answer
    assert "failed to record the turn" in caplog.text
    # the whole recording was one transaction, so nothing half-written is left behind
    assert [m.role for m in await repo.list_messages(pool, turn.conversation_id)] == [
        "user"
    ]


# --- trimming is recorded ---------------------------------------------------------------


async def test_a_trim_is_recorded_when_history_does_not_fit(pool):
    conv = await repo.create_conversation(pool)
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        await repo.add_message(pool, conv, role, "w" * 1500)
    settings = make_settings(
        max_context_tokens=2500
    )  # reply reserve leaves ~1500 for the prompt

    turn, events = await run_turn(
        pool, ScriptedFake(), "latest", conv, settings=settings
    )

    assert isinstance(events[-1], Done)
    (event,) = await repo.list_trim_events(pool, conv)
    assert event.request_id == turn.request_id
    assert event.messages_dropped > 0 and event.reason == "budget"
    assert event.dropped_through_message_id is not None
