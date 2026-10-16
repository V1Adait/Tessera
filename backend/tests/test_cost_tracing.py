"""Every call that costs money leaves a trace with its cost and where the number came from.

One group per kind of paid call (the JSON helper used by the LLM classifier, metrics, planner and
cell extractor; chat; summaries; embeddings; the Jev classifier), then the rules the database
enforces about cost, then the report. Needs the database (conftest) for the parts that store traces.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import asyncpg
import httpx
import pytest
from langchain_core.messages import HumanMessage

from backend.app import chat_service, summarise
from backend.app.costs import Price, PriceBook, set_price_book
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import TraceRecord
from backend.app.embeddings.embedder import OpenRouterEmbedder
from backend.app.main import app
from backend.app.query.classify import JevClassifier
from backend.app.query.llm_json import ModelCallError, ask_json
from backend.tests.fakes import (
    QueueFake,
    ScriptedFake,
    SlowFake,
    SummaryFake,
    make_settings,
)
from backend.tests.test_chat_service import run_turn
from backend.tests.test_summarise import seed, tight_settings

D = Decimal


def prices(model="m", prompt="0.001", completion="0.01"):
    set_price_book(PriceBook({model: Price(D(prompt), D(completion))}))


# --- the JSON helper (LLM classifier, metrics, planner, cell extractor) -----------------------


async def call_json(llm, kind="extract"):
    """Run ask_json and return (traces recorded, the error it raised or None)."""
    traces: list[TraceRecord] = []

    async def keep(t):
        traces.append(t)

    error = None
    try:
        await ask_json(
            llm,
            [HumanMessage(content="hi")],
            recorder=keep,
            kind=kind,
            provider="openrouter",
            model="m",
            prompt_version="v1",
        )
    except ModelCallError as exc:
        error = exc
    return traces, error


async def test_a_cost_the_provider_reported_is_recorded_as_provider_cost():
    traces, _ = await call_json(QueueFake(replies=["{}"], reported_cost=0.00042))

    [t] = traces
    assert (t.cost_usd, t.cost_source, t.tokens_in, t.tokens_out) == (
        D("0.00042"),
        "provider",
        50,
        10,
    )


async def test_without_a_reported_cost_it_is_computed_from_the_price_list():
    prices(prompt="0.00001", completion="0.0001")  # 50 in, 10 out

    [t], _ = await call_json(QueueFake(replies=["{}"]))

    assert (t.cost_usd, t.cost_source) == (D("0.0015"), "computed")


async def test_a_reported_cost_wins_over_the_price_list():
    prices(prompt="1", completion="1")

    [t], _ = await call_json(QueueFake(replies=["{}"], reported_cost=0.0002))

    assert (t.cost_usd, t.cost_source) == (D("0.0002"), "provider")


async def test_a_paid_call_with_no_known_price_is_unknown_not_free():
    [t], _ = await call_json(QueueFake(replies=["{}"]))

    assert t.cost_usd is None and t.cost_source == "unknown" and t.tokens_in == 50


async def test_a_failed_call_records_no_cost():
    [t], error = await call_json(QueueFake(replies=[RuntimeError("boom")]), kind="plan")

    assert error is not None
    assert t.status == "error" and t.cost_usd is None and t.cost_source is None


async def test_a_reply_that_is_not_json_was_still_paid_for():
    [t], error = await call_json(QueueFake(replies=["not json"], reported_cost=0.0003))

    assert error is not None  # the caller could not use it...
    assert t.status == "ok" and t.cost_usd == D(
        "0.0003"
    )  # ...but the provider billed it


# --- chat -------------------------------------------------------------------------------------


async def test_a_finished_chat_turn_is_priced_from_its_tokens(pool):
    prices("m", prompt="0.001", completion="0.01")  # ScriptedFake: 5 tokens in, 2 out

    turn, _ = await run_turn(pool, ScriptedFake(), "hi")

    (t,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert (t.cost_usd, t.cost_source) == (D("0.025"), "computed")


async def test_a_chat_model_with_no_known_price_is_recorded_as_unknown_cost(pool):
    turn, _ = await run_turn(pool, ScriptedFake(), "hi")

    (t,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert t.cost_usd is None and t.cost_source == "unknown"
    assert (t.tokens_in, t.tokens_out) == (5, 2)


async def test_a_chat_turn_stopped_part_way_is_unknown_because_it_was_still_billed(
    pool,
):
    turn = await chat_service.begin_turn(pool, None, "hi", uuid4())
    stream = chat_service.stream_turn(pool, make_settings(), SlowFake(), turn)
    async for event in stream:
        if getattr(event, "text", None):
            break
    await stream.aclose()

    traces = []
    for _ in range(100):  # recording after a disconnect happens in the background
        traces = await repo.list_traces(pool, conversation_id=turn.conversation_id)
        if traces:
            break
        await asyncio.sleep(0.03)
    (t,) = traces
    assert t.cost_usd is None and t.cost_source == "unknown"


async def test_a_turn_refused_before_the_model_has_no_cost_at_all(pool):
    settings = make_settings(max_context_tokens=60)

    turn, _ = await run_turn(pool, ScriptedFake(), "x" * 5000, settings=settings)

    (t,) = await repo.list_traces(pool, conversation_id=turn.conversation_id)
    assert t.status == "rejected" and t.cost_usd is None and t.cost_source is None


# --- summaries --------------------------------------------------------------------------------


async def test_a_summary_records_the_cost_the_provider_reported(pool):
    conv, _ = await seed(pool)

    await summarise.maybe_summarise(
        pool,
        tight_settings(),
        SummaryFake(reported_cost=0.00031),
        conv,
        tokens_used=2900,
    )

    (t,) = await repo.list_traces(pool, conversation_id=conv)
    assert t.kind == "summary" and (t.cost_usd, t.cost_source) == (
        D("0.00031"),
        "provider",
    )


async def test_a_summary_without_a_reported_cost_is_computed_from_the_price_list(pool):
    conv, _ = await seed(pool)
    settings = tight_settings()
    prices(
        settings.summary_model, prompt="0.00001", completion="0.0001"
    )  # 300 in, 40 out

    await summarise.maybe_summarise(
        pool, settings, SummaryFake(), conv, tokens_used=2900
    )

    (t,) = await repo.list_traces(pool, conversation_id=conv)
    assert (t.cost_usd, t.cost_source) == (D("0.007"), "computed")


async def test_a_summary_that_was_refused_was_still_paid_for(pool):
    conv, _ = await seed(pool)
    too_long = SummaryFake(reply="word " * 5000, reported_cost=0.0005)

    outcome = await summarise.maybe_summarise(
        pool, tight_settings(), too_long, conv, tokens_used=2900
    )

    (t,) = await repo.list_traces(pool, conversation_id=conv)
    assert outcome == "rejected" and t.status == "rejected"
    assert (t.cost_usd, t.cost_source) == (D("0.0005"), "provider")


# --- embeddings and the Jev classifier ---------------------------------------------------------


def embedder_with(handler, recorded):
    async def keep(t):
        recorded.append(t)

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://x.test/api/v1"
    )
    return OpenRouterEmbedder(http, model="m/embed", recorder=keep, backoff_s=0.0)


async def test_an_embedding_call_records_the_provider_cost():
    recorded: list[TraceRecord] = []
    reply = {
        "data": [{"index": 0, "embedding": [1.0]}],
        "usage": {"prompt_tokens": 7, "cost": 8.6e-07},
    }

    await embedder_with(lambda r: httpx.Response(200, json=reply), recorded).embed(
        ["a"]
    )

    [t] = recorded
    assert (t.kind, t.cost_usd, t.cost_source, t.tokens_in) == (
        "embed",
        D("8.6E-7"),
        "provider",
        7,
    )


async def test_an_embedding_call_the_provider_did_not_price_is_unknown():
    recorded: list[TraceRecord] = []
    reply = {"data": [{"index": 0, "embedding": [1.0]}], "usage": {"prompt_tokens": 7}}

    await embedder_with(lambda r: httpx.Response(200, json=reply), recorded).embed(
        ["a"]
    )

    [t] = recorded
    assert t.cost_usd is None and t.cost_source == "unknown"


async def test_a_rejected_embedding_call_records_no_cost():
    recorded: list[TraceRecord] = []

    with pytest.raises(Exception):  # noqa: B017
        await embedder_with(lambda r: httpx.Response(401), recorded).embed(["a"])

    [t] = recorded
    assert t.status == "error" and t.cost_usd is None and t.cost_source is None


def jev(handler, recorded):
    async def keep(t):
        recorded.append(t)

    return JevClassifier(
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://x.test/api"
        ),
        recorder=keep,
    )


JEV_REPLY = {
    "answers": {
        "type": {
            "type": "choice",
            "choice": "lookup",
            "confidence": 1.0,
            "probabilities": {},
        },
        "needs_decomposition": {"type": "noul", "noul": 0.2},
    },
}


async def test_a_jev_call_records_the_provider_cost():
    recorded: list[TraceRecord] = []
    body = {
        **JEV_REPLY,
        "usage": {"input_tokens": 489, "output_tokens": 92, "cost": 2.0538e-05},
    }

    await jev(lambda r: httpx.Response(200, json=body), recorded).classify(
        "What was revenue?"
    )

    [t] = recorded
    assert (t.kind, t.cost_usd, t.cost_source) == (
        "classify",
        D("0.000020538"),
        "provider",
    )


async def test_a_jev_call_without_a_cost_is_unknown():
    recorded: list[TraceRecord] = []
    body = {**JEV_REPLY, "usage": {"input_tokens": 489, "output_tokens": 92}}

    await jev(lambda r: httpx.Response(200, json=body), recorded).classify(
        "What was revenue?"
    )

    [t] = recorded
    assert t.cost_usd is None and t.cost_source == "unknown"


# --- what the database enforces -------------------------------------------------------------------


def trace(**kw) -> TraceRecord:
    return TraceRecord(
        request_id=uuid4(),
        kind=kw.pop("kind", "chat"),
        status=kw.pop("status", "ok"),
        **kw,
    )


async def test_a_tiny_cost_is_stored_exactly(pool):
    await repo.record_trace(
        pool, trace(cost_usd=D("0.00000086"), cost_source="provider")
    )

    (t,) = await repo.list_traces(pool)
    assert t.cost_usd == D("0.0000008600")  # it used to round to 0.000001


async def test_a_thousand_tiny_costs_add_up_exactly(pool):
    for _ in range(1000):
        await repo.record_trace(
            pool, trace(kind="embed", cost_usd=D("0.00000086"), cost_source="provider")
        )

    lines = await repo.cost_summary(pool, datetime.now(tz=UTC) - timedelta(hours=1))

    assert sum(line.cost_usd for line in lines) == D("0.00086")


async def test_a_cost_must_say_where_it_came_from(pool):
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.record_trace(pool, trace(cost_usd=D("0.1")))  # a number from nowhere


async def test_a_source_without_a_cost_is_refused_unless_it_is_unknown(pool):
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.record_trace(pool, trace(cost_source="provider"))
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.record_trace(pool, trace(cost_usd=D("0.1"), cost_source="unknown"))
    await repo.record_trace(
        pool, trace(cost_source="unknown")
    )  # a paid call, cost not known


async def test_a_negative_cost_is_refused(pool):
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.record_trace(pool, trace(cost_usd=D("-1"), cost_source="provider"))


# --- the report --------------------------------------------------------------------------------


async def seed_costs(pool):
    await repo.record_trace(
        pool,
        trace(
            kind="extract",
            model="a",
            cost_usd=D("0.5"),
            cost_source="provider",
            tokens_in=100,
            tokens_out=10,
        ),
    )
    await repo.record_trace(
        pool,
        trace(
            kind="extract",
            model="a",
            cost_usd=D("0.25"),
            cost_source="computed",
            tokens_in=50,
            tokens_out=5,
        ),
    )
    await repo.record_trace(
        pool,
        trace(
            kind="extract", model="a", cost_source="unknown", tokens_in=10, tokens_out=1
        ),
    )
    await repo.record_trace(
        pool,
        trace(
            kind="embed",
            model="e",
            cost_usd=D("0.01"),
            cost_source="provider",
            tokens_in=999,
        ),
    )
    await repo.record_trace(
        pool, trace(kind="chat", status="rejected")
    )  # never reached the model


async def test_the_summary_adds_up_by_kind_and_model_biggest_first(pool):
    await seed_costs(pool)

    lines = await repo.cost_summary(pool, datetime.now(tz=UTC) - timedelta(hours=1))

    assert [(line.kind, line.model) for line in lines] == [
        ("extract", "a"),
        ("embed", "e"),
        ("chat", None),
    ]
    extract = lines[0]
    assert (extract.calls, extract.cost_usd) == (3, D("0.75"))
    assert (extract.unknown_cost_calls, extract.estimated_calls) == (1, 1)
    assert (extract.tokens_in, extract.tokens_out) == (160, 16)
    assert (
        lines[2].unknown_cost_calls == 0
    )  # a refused request is not a paid call with no cost


async def test_the_summary_only_counts_the_period_asked_for(pool):
    await seed_costs(pool)

    assert (
        await repo.cost_summary(pool, datetime.now(tz=UTC) + timedelta(hours=1)) == []
    )


@pytest.fixture
async def client(pool):
    app.dependency_overrides[get_pool] = lambda: pool
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_the_costs_route_reports_totals_and_flags_a_lower_bound(client, pool):
    await seed_costs(pool)

    body = (await client.get("/costs", params={"hours": 24})).json()

    assert D(body["total_usd"]) == D("0.76")
    assert body["calls"] == 5
    assert body["unknown_cost_calls"] == 1 and body["estimated_calls"] == 1
    assert body["lines"][0]["kind"] == "extract"


async def test_the_costs_route_validates_the_period(client):
    assert (await client.get("/costs", params={"hours": 0})).status_code == 422
    assert (await client.get("/costs", params={"hours": 99999})).status_code == 422
    assert (await client.get("/costs")).status_code == 200
