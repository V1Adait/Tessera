"""The OpenRouter embedder, against a mock HTTP transport: no network, no key."""

import json

import httpx
import pytest

from backend.app.db.records import TraceRecord
from backend.app.embeddings.embedder import (
    EmbeddingError,
    OpenRouterEmbedder,
    check_vectors,
    embed_in_batches,
)


def reply(vectors):
    return {
        "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)],
        "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 0.0000001},
    }


async def _ignore(trace: TraceRecord) -> None:
    return None


def embedder(handler, traces=None):
    async def record(trace: TraceRecord) -> None:
        traces.append(trace)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test/api/v1"
    )
    return OpenRouterEmbedder(
        client,
        model="m/embed",
        recorder=_ignore if traces is None else record,
        backoff_s=0.0,
    )


async def test_it_sends_the_model_and_texts_and_returns_vectors_in_order():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), path=request.url.path)
        return httpx.Response(200, json=reply([[1.0, 0.0], [0.0, 1.0]]))

    vectors = await embedder(handler).embed(["a", "b"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert seen["model"] == "m/embed" and seen["input"] == ["a", "b"]
    assert seen["path"].endswith("/embeddings")


async def test_vectors_are_put_back_in_input_order_even_if_the_reply_is_shuffled():
    def handler(request):
        body = {
            "data": [
                {"index": 1, "embedding": [2.0]},
                {"index": 0, "embedding": [1.0]},
            ]
        }
        return httpx.Response(200, json=body)

    assert await embedder(handler).embed(["a", "b"]) == [[1.0], [2.0]]


async def test_an_empty_list_makes_no_call():
    def handler(request):
        raise AssertionError("no request expected")

    assert await embedder(handler).embed([]) == []


async def test_a_rate_limit_is_retried_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=reply([[1.0]]))

    assert await embedder(handler).embed(["a"]) == [[1.0]]
    assert len(attempts) == 3


async def test_it_gives_up_after_three_attempts_with_unavailable():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503)

    with pytest.raises(EmbeddingError) as caught:
        await embedder(handler).embed(["a"])

    assert caught.value.code == "unavailable" and len(attempts) == 3


async def test_a_network_failure_counts_as_unavailable():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(EmbeddingError) as caught:
        await embedder(handler).embed(["a"])

    assert caught.value.code == "unavailable"


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "auth"), (403, "auth"), (400, "bad_request"), (404, "bad_request")],
)
async def test_client_errors_are_not_retried_and_hide_provider_detail(status, code):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(
            status, json={"error": {"message": "secret provider detail"}}
        )

    with pytest.raises(EmbeddingError) as caught:
        await embedder(handler).embed(["a"])

    assert caught.value.code == code and len(attempts) == 1
    assert "secret" not in caught.value.message


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},  # wrong count
        {"data": [{"index": 0, "embedding": []}]},  # empty vector
        {"data": [{"index": 0, "embedding": [1.0, "x"]}]},  # not numbers
        {"data": [{"index": 0, "embedding": [float("nan")]}]},  # NaN
        {"oops": 1},  # wrong shape
    ],
)
def test_unusable_replies_are_rejected(body):
    with pytest.raises(EmbeddingError) as caught:
        check_vectors(body, 1)

    assert caught.value.code == "bad_output"


def test_vectors_of_different_lengths_are_rejected():
    body = {
        "data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [1.0, 2.0]},
        ]
    }
    with pytest.raises(EmbeddingError):
        check_vectors(body, 2)


async def test_a_reply_that_is_not_json_is_bad_output():
    def handler(request):
        return httpx.Response(200, text="<html>nope</html>")

    with pytest.raises(EmbeddingError) as caught:
        await embedder(handler).embed(["a"])

    assert caught.value.code == "bad_output"


async def test_every_call_leaves_a_trace_with_tokens_and_cost():
    traces: list[TraceRecord] = []

    def handler(request):
        return httpx.Response(200, json=reply([[1.0]]))

    await embedder(handler, traces).embed(["a"])

    [trace] = traces
    assert (trace.kind, trace.status, trace.model) == ("embed", "ok", "m/embed")
    assert trace.tokens_in == 7 and float(trace.cost_usd or 0) > 0
    assert trace.latency_ms is not None


async def test_a_failed_call_is_traced_with_its_code_and_no_secrets():
    traces: list[TraceRecord] = []

    def handler(request):
        return httpx.Response(401, text="bad key sk-secret")

    with pytest.raises(EmbeddingError):
        await embedder(handler, traces).embed(["a"])

    [trace] = traces
    assert trace.status == "error" and trace.error_code == "auth"
    assert "sk-secret" not in trace.model_dump_json()


async def test_batching_splits_the_texts_and_keeps_the_order():
    sizes = []

    def handler(request):
        texts = json.loads(request.content)["input"]
        sizes.append(len(texts))
        return httpx.Response(200, json=reply([[float(t)] for t in texts]))

    vectors = await embed_in_batches(embedder(handler), [str(i) for i in range(5)], 2)

    assert sizes == [2, 2, 1]
    assert vectors == [[float(i)] for i in range(5)]
