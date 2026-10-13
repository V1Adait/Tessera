"""Query handling, start to finish:  question in, QueryPlan out.

    workspace entities  ->  route (classify + confidence gate)  ->  slots (entities, periods)
        chat        -> no retrieval
        evaluative  -> RubricPlanner (LLM picks rubric dimensions)
        otherwise   -> metrics (LLM + code check) -> template expansion (code)
    -> validate_plan  -> status: ready | needs_confirmation | needs_clarification | no_retrieval

Nothing here retrieves or answers. It only decides WHAT would have to be looked up.
"""

import logging
from datetime import date
from uuid import UUID

import asyncpg
import httpx

from backend.app.config import Settings
from backend.app.db import repository as repo
from backend.app.llm import get_llm
from backend.app.query.classify import (
    Classifier,
    JevClassifier,
    LLMClassifier,
    make_jev_client,
)
from backend.app.query.metrics import MetricExtractor
from backend.app.query.models import EntityRef, QueryPlan, Slots
from backend.app.query.plan import Expansion, expand, validate_plan
from backend.app.query.planner import RubricPlanner, build_tasks, default_pairs
from backend.app.query.router import route
from backend.app.query.slots import find_entities, find_periods
from backend.app.tracing import Recorder, db_recorder

logger = logging.getLogger(__name__)

_NEEDS_METRICS = {"multi_field", "comparison", "trend", "aggregation"}


class QueryService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        *,
        classifiers: list[Classifier],
        second_opinion: Classifier | None = None,
        extractor: MetricExtractor | None = None,
        planner: RubricPlanner | None = None,
        closers: list[httpx.AsyncClient] | None = None,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._classifiers = classifiers
        self._second_opinion = second_opinion
        self._extractor = extractor
        self._planner = planner
        self._closers = closers or []

    async def close(self) -> None:
        for client in self._closers:
            await client.aclose()

    async def plan(
        self, workspace_id: UUID, question: str, today: date | None = None
    ) -> QueryPlan:
        """Raises repo.WorkspaceNotFoundError for an unknown workspace; nothing else."""
        if await repo.get_workspace(self._pool, workspace_id) is None:
            raise repo.WorkspaceNotFoundError(str(workspace_id))
        records = await repo.list_entities(self._pool, workspace_id)
        refs = [EntityRef(id=e.id, name=e.name) for e in records]

        routing = await route(
            question,
            self._classifiers,
            threshold=self._settings.router_confidence_threshold,
            second_opinion=self._second_opinion,
        )
        notes: list[str] = []
        slots = Slots(
            entities=find_entities(question, records),
            periods=find_periods(question, today),
        )
        plan = QueryPlan(
            question=question,
            question_type=routing.question_type,
            status="ready",
            routing=routing,
            slots=slots,
        )

        if routing.question_type == "chat":
            plan.status = "no_retrieval"
            return plan

        expansion = await self._expand(plan, refs, notes)
        plan.tasks = expansion.tasks
        plan.combine = expansion.combine
        plan.notes = notes + expansion.notes
        if expansion.clarification:
            plan.status = "needs_clarification"
            plan.clarification = expansion.clarification
            return plan

        plan.status = validate_plan(
            plan,
            soft_cap=self._settings.plan_soft_cap,
            hard_cap=self._settings.plan_hard_cap,
        )
        return plan

    async def _expand(
        self, plan: QueryPlan, workspace: list[EntityRef], notes: list[str]
    ) -> Expansion:
        slots, question = plan.slots, plan.question

        if plan.question_type == "evaluative":
            if not slots.entities:
                return Expansion(clarification="Which company is this about?")
            if self._planner is None:
                notes.append("no planner model available: used the full rubric")
                return Expansion(
                    tasks=build_tasks(default_pairs(), slots.entities), combine="assess"
                )
            result = await self._planner.plan(question, slots.entities)
            return Expansion(tasks=result.tasks, combine="assess", notes=result.notes)

        if plan.question_type in _NEEDS_METRICS:
            await self._add_metrics(plan, notes)
        return expand(plan.question_type, question, slots, workspace)

    async def _add_metrics(self, plan: QueryPlan, notes: list[str]) -> None:
        if self._extractor is None:
            notes.append("no metric extractor available: metrics left unnamed")
            return
        found = await self._extractor.extract(plan.question)
        plan.slots.metrics = found.metrics
        if found.failed:
            notes.append("metric extraction failed: metrics left unnamed")
        if found.dropped:
            notes.append(f"dropped metric(s) not in the question: {found.dropped}")


def build_query_service(
    pool: asyncpg.Pool, settings: Settings, recorder: Recorder | None = None
) -> QueryService | None:
    """Assemble the real service from the settings, or None if no classifier can be built
    (no API keys). Jev needs the OpenRouter key; the LLM pieces need the router provider's."""
    recorder = recorder or db_recorder(pool)
    classifiers: list[Classifier] = []
    closers: list[httpx.AsyncClient] = []

    if settings.openrouter_api_key:
        client = make_jev_client(settings.openrouter_api_key, settings.router_timeout_s)
        closers.append(client)
        classifiers.append(
            JevClassifier(client, model=settings.router_jev_model, recorder=recorder)
        )

    llm_classifier: LLMClassifier | None = None
    extractor: MetricExtractor | None = None
    planner: RubricPlanner | None = None
    provider, model = settings.router_provider, settings.router_llm_model
    if settings.key_for(provider):
        small = get_llm(
            settings,
            provider=provider,
            model=model,
            max_tokens=settings.router_max_tokens,
            temperature=0,
        )
        big = get_llm(
            settings,
            provider=provider,
            model=model,
            max_tokens=settings.planner_max_tokens,
            temperature=0,
        )
        llm_classifier = LLMClassifier(
            small, provider=provider, model=model, recorder=recorder
        )
        classifiers.append(llm_classifier)
        extractor = MetricExtractor(
            small, provider=provider, model=model, recorder=recorder
        )
        planner = RubricPlanner(
            big,
            provider=provider,
            model=model,
            max_dimensions=settings.planner_max_dimensions,
            recorder=recorder,
        )

    if not classifiers:
        logger.warning("no key for a query classifier: question planning is off")
        return None
    return QueryService(
        pool,
        settings,
        classifiers=classifiers,
        second_opinion=llm_classifier,
        extractor=extractor,
        planner=planner,
        closers=closers,
    )
