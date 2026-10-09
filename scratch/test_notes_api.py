import pytest
from fastapi.testclient import TestClient

from scratch.notes_api import app, history

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_history():
    history.clear()  # every test starts with an empty history


# --- tests for the example as given (these already pass) ---------------------------------


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_remembers_it():
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200
    assert r.json() == {"question": "hi", "answer": "You asked: hi"}
    assert client.get("/history").json() == [r.json()]


def test_missing_field_is_422():
    assert client.post("/ask", json={}).status_code == 422


# --- exercise targets: remove the skip marker, watch it fail, make it pass ---------------

exercise = pytest.mark.skip(
    reason="exercise: write the code in notes_api.py, then remove this"
)


def test_ex1_empty_or_huge_question_is_422():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 500}).status_code == 200


@exercise
def test_ex2_get_one_answer_by_position():
    client.post("/ask", json={"question": "first"})
    client.post("/ask", json={"question": "second"})
    assert client.get("/history/0").json()["question"] == "first"
    assert client.get("/history/1").json()["question"] == "second"
    missing = client.get("/history/5")
    assert missing.status_code == 404
    assert missing.json()["detail"]


@exercise
def test_ex3_limit_returns_last_n():
    for word in ["a", "b", "c", "d"]:
        client.post("/ask", json={"question": word})
    assert [x["question"] for x in client.get("/history?limit=2").json()] == ["c", "d"]
    assert len(client.get("/history").json()) == 4
    assert client.get("/history?limit=0").status_code == 422


def test_ex4_delete_clears_history():
    client.post("/ask", json={"question": "x"})
    r = client.delete("/history")
    assert r.status_code == 204
    assert client.get("/history").json() == []


def test_ex5_answer_streams_word_by_word():
    with client.stream("POST", "/ask/stream", json={"question": "hi"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert body == "data: You\n\ndata: asked:\n\ndata: hi\n\ndata: [DONE]\n\n"
