"""Query handling end to end: question -> routing -> slots -> plan, against a real (throwaway)
database for the workspace and entities. Classifiers and models are scripted fakes."""

import json
from collections.abc import AsyncIterator
from datetime import date
from uuid import uuid4

import httpx
import pytest

from backend.app.api_query import get_query_service
from backend.app.db import repository as repo
from backend.app.db.records import TraceRecord
from backend.app.main import app
from backend.app.query.metrics import MetricExtractor
from backend.app.query.models import Prediction
from backend.app.query.planner import RubricPlanner
from backend.app.query.service import QueryService
from backend.tests.fakes import QueueFake, make_settings

TODAY = date(2026, 10, 13)


class Fixed:
    """A classifier that always answers the same."""

    def __init__(self, label, confidence=0.95):
        self.name = "fixed"
        self._p = Prediction(label=label, confidence=confidence, classifier="fixed")

    async def classify(self, question: str) -> Prediction:
        return self._p


def metrics_reply(*metrics: str) -> str:
    return json.dumps({"metrics": list(metrics)})


def planner_reply(*dims: str) -> str:
    tasks = [
        {"dimension": d, "question": f"What about {d} for {{entity}}?"} for d in dims
    ]
    return json.dumps({"tasks": tasks})


async def make_service(pool, label, *, metrics=None, plan=None, settings=None, **kw):
    extractor = MetricExtractor(
        QueueFake(replies=[metrics or metrics_reply()]), provider="p", model="m"
    )
    planner = RubricPlanner(
        QueueFake(replies=[plan or planner_reply("growth")]),
        provider="p",
        model="m",
        max_dimensions=6,
    )
    return QueryService(
        pool,
        settings or make_settings(),
        classifiers=[Fixed(label, **kw)],
        extractor=extractor,
        planner=planner,
    )


async def workspace_with(pool, *names: str):
    ws = await repo.create_workspace(pool, f"w-{uuid4().hex[:6]}")
    for name in names:
        await repo.create_entity(pool, ws.id, name)
    return ws


async def test_chat_needs_no_retrieval(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(pool, "chat")

    plan = await service.plan(ws.id, "thanks!", TODAY)

    assert plan.status == "no_retrieval" and plan.tasks == []


async def test_a_lookup_becomes_one_task_scoped_to_the_entity(pool):
    ws = await workspace_with(pool, "Acme", "Beta")
    service = await make_service(pool, "lookup")

    plan = await service.plan(ws.id, "What was Acme's FY24 revenue?", TODAY)

    assert plan.status == "ready"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].entity and plan.tasks[0].entity.name == "Acme"
    assert plan.tasks[0].period and plan.tasks[0].period.year == 2024
    assert plan.slots.entities[0].name == "Acme"


async def test_a_lookup_does_not_spend_a_call_on_metrics(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(pool, "lookup")
    assert service._extractor is not None

    await service.plan(ws.id, "What was Acme's revenue?", TODAY)

    assert service._extractor._llm.seen == []  # type: ignore[attr-defined]


async def test_a_comparison_gets_one_task_per_company_and_the_verified_metric(pool):
    ws = await workspace_with(pool, "Acme", "Beta", "Gamma")
    service = await make_service(
        pool, "comparison", metrics=metrics_reply("operating margin")
    )

    plan = await service.plan(ws.id, "Acme vs Beta on operating margin", TODAY)

    assert plan.status == "ready" and plan.combine == "compare"
    assert [t.entity.name for t in plan.tasks if t.entity] == ["Acme", "Beta"]
    assert plan.slots.metrics == ["operating margin"]


async def test_an_invented_metric_is_dropped_and_noted(pool):
    ws = await workspace_with(pool, "Acme", "Beta")
    service = await make_service(
        pool, "comparison", metrics=metrics_reply("operating margin", "ebitda")
    )

    plan = await service.plan(ws.id, "Acme vs Beta on operating margin", TODAY)

    assert plan.slots.metrics == ["operating margin"]
    assert any("dropped metric" in note for note in plan.notes)


async def test_a_failed_metric_call_does_not_fail_the_plan(pool):
    ws = await workspace_with(pool, "Acme", "Beta")
    service = await make_service(pool, "comparison")
    service._extractor = MetricExtractor(  # type: ignore[assignment]
        QueueFake(replies=[RuntimeError("down")]), provider="p", model="m"
    )

    plan = await service.plan(ws.id, "Acme vs Beta on margin", TODAY)

    assert plan.status == "ready" and len(plan.tasks) == 2
    assert any("metric extraction failed" in note for note in plan.notes)


async def test_a_trend_over_explicit_years_has_a_task_per_year(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(pool, "trend")

    plan = await service.plan(
        ws.id, "How has Acme's leverage changed 2022-2024?", TODAY
    )

    assert [t.period.year for t in plan.tasks if t.period] == [2022, 2023, 2024]
    assert plan.combine == "trend"


async def test_a_question_about_no_known_company_asks_which(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(pool, "lookup")

    plan = await service.plan(ws.id, "What was Zed's revenue?", TODAY)

    assert plan.status == "needs_clarification"
    assert plan.clarification and plan.tasks == []


async def test_aggregation_runs_over_every_entity_in_the_workspace(pool):
    ws = await workspace_with(pool, "Acme", "Beta", "Gamma")
    service = await make_service(pool, "aggregation")

    plan = await service.plan(ws.id, "Which companies breach covenants?", TODAY)

    assert plan.status == "ready" and len(plan.tasks) == 3 and plan.combine == "filter"


async def test_a_big_aggregation_needs_confirmation_before_it_runs(pool):
    ws = await workspace_with(pool, *[f"Company{i}" for i in range(10)])
    service = await make_service(pool, "aggregation")

    plan = await service.plan(ws.id, "Which companies breach covenants?", TODAY)

    assert plan.status == "needs_confirmation" and len(plan.tasks) == 10


async def test_too_many_tasks_are_refused_by_the_hard_cap(pool):
    from backend.app.query.plan import PlanError

    ws = await workspace_with(pool, *[f"Company{i}" for i in range(6)])
    settings = make_settings()
    settings.plan_hard_cap = 5
    service = await make_service(pool, "aggregation", settings=settings)

    with pytest.raises(PlanError):
        await service.plan(ws.id, "Which companies breach covenants?", TODAY)


async def test_an_open_question_uses_the_rubric_planner(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(
        pool, "evaluative", plan=planner_reply("growth", "risks")
    )

    plan = await service.plan(ws.id, "Is Acme a good investment?", TODAY)

    assert plan.status == "ready" and plan.combine == "assess"
    assert [t.dimension for t in plan.tasks] == ["growth", "risks"]


async def test_an_open_question_without_a_company_asks_which(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(pool, "evaluative")

    plan = await service.plan(ws.id, "Is it a good investment?", TODAY)

    assert plan.status == "needs_clarification"


async def test_a_low_confidence_question_is_planned_as_evaluative(pool):
    ws = await workspace_with(pool, "Acme")
    service = await make_service(
        pool, "trend", confidence=0.4, plan=planner_reply("growth")
    )

    plan = await service.plan(ws.id, "Acme's debt situation", TODAY)

    assert plan.question_type == "evaluative"
    assert plan.routing and plan.routing.low_confidence


async def test_an_unknown_workspace_is_reported(pool):
    service = await make_service(pool, "lookup")

    with pytest.raises(repo.WorkspaceNotFoundError):
        await service.plan(uuid4(), "q", TODAY)


async def test_entities_from_another_workspace_are_never_matched(pool):
    ours = await workspace_with(pool, "Acme")
    await workspace_with(pool, "Secretco")
    service = await make_service(pool, "lookup")

    plan = await service.plan(ours.id, "What was Secretco's revenue?", TODAY)

    assert plan.status == "needs_clarification"


# --- the HTTP route -----------------------------------------------------------------------


@pytest.fixture
async def client(pool) -> AsyncIterator[httpx.AsyncClient]:
    service = await make_service(pool, "lookup")
    app.dependency_overrides[get_query_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_http_plan_returns_the_plan(client, pool):
    ws = await workspace_with(pool, "Acme")

    response = await client.post(
        f"/workspaces/{ws.id}/query/plan", json={"question": "Acme revenue?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready" and body["tasks"][0]["entity"]["name"] == "Acme"


async def test_http_unknown_workspace_is_404(client):
    response = await client.post(
        f"/workspaces/{uuid4()}/query/plan", json={"question": "hi"}
    )

    assert response.status_code == 404


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 2001}, {}])
async def test_http_bad_bodies_are_422(client, pool, body):
    ws = await workspace_with(pool, "Acme")

    response = await client.post(f"/workspaces/{ws.id}/query/plan", json=body)

    assert response.status_code == 422


async def test_http_without_a_configured_service_is_503(pool):
    transport = httpx.ASGITransport(app=app)
    app.state.query = None
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.post(
            f"/workspaces/{uuid4()}/query/plan", json={"question": "hi"}
        )

    assert response.status_code == 503


async def test_traces_can_be_stored_with_the_new_kinds(pool):
    for kind in ("classify", "extract", "plan"):
        await repo.record_trace(
            pool, TraceRecord(request_id=uuid4(), kind=kind, status="ok")
        )
