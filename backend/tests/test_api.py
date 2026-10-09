"""Tests for the HTTP layer. The real model is replaced by a fake, so no key and no cost."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app import api, pipeline
from backend.app.config import get_settings
from backend.app.llm import get_default_llm
from backend.app.main import app
from backend.tests.fakes import ScriptedFake, make_settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = make_settings
    # a function returning an instance, NOT the class: FastAPI would read a class's fields
    # as request parameters
    app.dependency_overrides[get_default_llm] = lambda: ScriptedFake()
    yield TestClient(
        app
    )  # no `with`: the startup check is skipped, it needs a real .env
    app.dependency_overrides.clear()


def body(text: str = "hi") -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def parse_sse(raw: str) -> list[dict]:
    """'data: {...}\\n\\n' blocks -> list of dicts."""
    blocks = [b for b in raw.split("\n\n") if b]
    assert all(b.startswith("data: ") for b in blocks)
    return [json.loads(b.removeprefix("data: ")) for b in blocks]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_streams_deltas_then_done_as_sse(client):
    with client.stream("POST", "/chat", json=body()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse("".join(r.iter_text()))

    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "hello there"
    assert events[-1]["usage"] == {"tokens_in": 5, "tokens_out": 2}


def test_request_id_header_matches_the_done_event(client):
    with client.stream("POST", "/chat", json=body()) as r:
        header_id = r.headers["x-request-id"]
        events = parse_sse("".join(r.iter_text()))

    assert events[-1]["request_id"] == header_id


def test_malformed_body_is_rejected_with_422(client):
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"messages": []}).status_code == 422
    bad_role = {"messages": [{"role": "robot", "content": "hi"}]}
    assert client.post("/chat", json=bad_role).status_code == 422


def test_context_too_large_is_an_error_event_in_the_stream(client):
    app.dependency_overrides[get_settings] = lambda: make_settings(
        max_context_tokens=5000
    )

    with client.stream("POST", "/chat", json=body("x" * 60_000)) as r:
        assert (
            r.status_code == 200
        )  # headers were already sent; the failure is in the stream
        events = parse_sse("".join(r.iter_text()))

    assert [e["type"] for e in events] == ["error"]
    assert events[0]["code"] == "context_too_large"


def test_a_crashing_pipeline_still_ends_with_an_error_event(client, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")
        yield  # makes this an async generator

    monkeypatch.setattr(pipeline, "run", boom)

    with client.stream("POST", "/chat", json=body()) as r:
        raw = "".join(r.iter_text())

    events = parse_sse(raw)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "unknown"
    assert "secret" not in raw  # internal detail never reaches the client


def test_cors_allows_the_ui_origin(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_allow_other_origins(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in r.headers


def test_sse_helper_format():
    from backend.app.models import Delta

    assert api.sse(Delta(text="hi")) == 'data: {"type":"delta","text":"hi"}\n\n'
