"""Tests for pipeline.run."""

from uuid import uuid4

from backend.app import pipeline
from backend.app.models import ChatRequest, Delta, Done, ErrorEvent, Message
from backend.tests.fakes import ScriptedFake, make_settings, request


async def test_happy_path_streams_deltas_then_done():
    events = [e async for e in pipeline.run(request(), make_settings(), ScriptedFake())]

    assert all(isinstance(e, Delta) for e in events[:-1])
    assert isinstance(events[-1], Done)
    assert "".join(e.text for e in events[:-1] if isinstance(e, Delta)) == "hello there"


async def test_model_receives_the_built_context_not_the_raw_request():
    fake = ScriptedFake()
    _ = [
        e
        async for e in pipeline.run(request("what is revenue?"), make_settings(), fake)
    ]

    prompt = fake.seen[0]
    assert prompt[0].type == "system"  # our system prompt comes first
    assert prompt[-1].content == "what is revenue?"  # the new message comes last


async def test_request_id_is_passed_through():
    rid = uuid4()
    events = [
        e async for e in pipeline.run(request(), make_settings(), ScriptedFake(), rid)
    ]

    assert events[-1].request_id == rid  # type: ignore[union-attr]


async def test_context_too_large_becomes_one_error_event_not_an_exception():
    big = request("x" * 60_000)
    events = [
        e
        async for e in pipeline.run(
            big, make_settings(max_context_tokens=5000), ScriptedFake()
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "context_too_large"
    assert events[0].retryable is False


async def test_input_and_output_check_slots_are_called(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "check_input", lambda req, s: calls.append("in"))
    monkeypatch.setattr(pipeline, "check_output", lambda e: calls.append("out") or e)

    _ = [e async for e in pipeline.run(request(), make_settings(), ScriptedFake())]

    assert calls[0] == "in"
    assert calls.count("out") == 3  # two Deltas and one Done


async def test_conversation_not_ending_with_user_message_is_a_bad_request_event():
    bad = ChatRequest(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
    )
    events = [e async for e in pipeline.run(bad, make_settings(), ScriptedFake())]

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "bad_request"


async def test_error_event_carries_the_given_request_id():
    rid = uuid4()
    big = request("x" * 60_000)
    events = [
        e
        async for e in pipeline.run(
            big, make_settings(max_context_tokens=5000), ScriptedFake(), rid
        )
    ]

    assert isinstance(events[0], ErrorEvent)
    assert events[0].request_id == rid
