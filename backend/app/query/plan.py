"""Turn a classified question into sub-tasks, by templates, in code. No model is involved here.

    lookup / multi_field / comparison / trend / aggregation   -> expand()   (this file)
    evaluative                                                -> planner.py (an LLM picks from the rubric)
    chat                                                      -> no tasks

Every function returns an `Expansion`. When a slot the plan needs is missing we do NOT guess:
the Expansion carries a `clarification` question for the user instead of tasks.

`validate_plan` is the safety net that runs over EVERY plan, whoever built it.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from backend.app.query.models import (
    Combine,
    EntityRef,
    Period,
    PlanStatus,
    QueryPlan,
    Slots,
    SubTask,
)

_VERSUS = re.compile(
    r"\b(?:vs\.?|versus|against|compared (?:to|with)|compare .+ (?:to|with)|than)\b",
    re.IGNORECASE,
)


@dataclass
class Expansion:
    tasks: list[SubTask] = field(default_factory=list)
    combine: Combine = "none"
    clarification: str | None = None
    notes: list[str] = field(default_factory=list)


class PlanError(ValueError):
    """A plan that must not run. `reasons` lists every problem found, not just the first."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


# --- building tasks ---------------------------------------------------------------------


def _words(metric: str | None, entity: EntityRef, period: Period | None) -> str:
    what = metric if metric else "the figures asked about"
    return f"{what} for {entity.name}" + (f" in {period.label}" if period else "")


def _tasks(
    question: str,
    entities: list[EntityRef],
    periods: Sequence[Period | None],
    metrics: Sequence[str | None],
) -> list[SubTask]:
    """One lookup per (entity, period, metric). With no metric named, the task carries the
    user's own question: we cannot say it better than they did."""
    tasks: list[SubTask] = []
    for entity in entities:
        for period in periods:
            for metric in metrics:
                tasks.append(
                    SubTask(
                        id=f"t{len(tasks) + 1}",
                        kind="lookup",
                        question=_words(metric, entity, period) if metric else question,
                        entity=entity,
                        period=period,
                        metric=metric,
                    )
                )
    return tasks


def _ask(text: str) -> Expansion:
    return Expansion(clarification=text)


# --- one template per class -------------------------------------------------------------


def expand_lookup(question: str, slots: Slots) -> Expansion:
    if not slots.entities:
        return _ask("Which company is this about?")
    tasks = _tasks(
        question, slots.entities, slots.periods or [None], slots.metrics[:1] or [None]
    )
    if len(tasks) == 1:
        return Expansion(tasks=tasks)
    note = "the question names several entities or periods, so it was split per one"
    return Expansion(tasks=tasks, combine="compare", notes=[note])


def expand_multi_field(question: str, slots: Slots) -> Expansion:
    if not slots.entities:
        return _ask("Which company is this about?")
    if not slots.metrics:
        return _ask(
            "Which facts do you want? Name them, for example revenue and net debt."
        )
    tasks = _tasks(
        question, slots.entities, slots.periods or [None], list(slots.metrics)
    )
    return Expansion(tasks=tasks)


def expand_comparison(question: str, slots: Slots) -> Expansion:
    if not slots.entities:
        return _ask("Which companies should I compare?")
    if len(slots.entities) == 1:
        if _VERSUS.search(question):
            # "Acme vs Zed" where Zed is not in the workspace: do not quietly drop it.
            only = slots.entities[0].name
            return _ask(
                f"I only found {only} in this workspace. Which other company should I compare it with?"
            )
        # Parts of one company ("which segment grew fastest"): we cannot name the parts
        # without reading the documents, so it stays one task for the retrieval step.
        tasks = _tasks(question, slots.entities, slots.periods[:1] or [None], [None])
        note = "comparison inside one entity: not split into parts before retrieval"
        return Expansion(tasks=tasks, combine="compare", notes=[note])
    tasks = _tasks(
        question,
        slots.entities,
        slots.periods[:1] or [None],
        slots.metrics[:1] or [None],
    )
    return Expansion(tasks=tasks, combine="compare")


def expand_trend(question: str, slots: Slots) -> Expansion:
    if not slots.entities:
        return _ask("Which company is this about?")
    metric: Sequence[str | None] = slots.metrics[:1] or [None]
    if len(slots.periods) >= 2:
        tasks = _tasks(question, slots.entities, list(slots.periods), metric)
        return Expansion(tasks=tasks, combine="trend")
    tasks = _tasks(question, slots.entities, [None], metric)
    note = "no period range given: the whole series in the documents will be read"
    return Expansion(tasks=tasks, combine="trend", notes=[note])


def expand_aggregation(
    question: str, slots: Slots, workspace_entities: list[EntityRef]
) -> Expansion:
    notes: list[str] = []
    entities = slots.entities
    if not entities:
        if not workspace_entities:
            return _ask("There are no companies in this workspace yet. Add some first.")
        entities = workspace_entities
        notes.append(
            "no company named: the question runs over every entity in the workspace"
        )
    tasks = _tasks(
        question, entities, slots.periods[:1] or [None], slots.metrics[:1] or [None]
    )
    return Expansion(tasks=tasks, combine="filter", notes=notes)


def expand(
    question_type: str,
    question: str,
    slots: Slots,
    workspace_entities: list[EntityRef],
) -> Expansion:
    """The template for a class. Evaluative and chat are handled by the caller."""
    match question_type:
        case "lookup":
            return expand_lookup(question, slots)
        case "multi_field":
            return expand_multi_field(question, slots)
        case "comparison":
            return expand_comparison(question, slots)
        case "trend":
            return expand_trend(question, slots)
        case "aggregation":
            return expand_aggregation(question, slots, workspace_entities)
    raise ValueError(f"no template for question type {question_type!r}")


# --- the safety net ---------------------------------------------------------------------


def _cycle(tasks: list[SubTask]) -> bool:
    """True if `depends_on` loops (Kahn's algorithm: peel off tasks with nothing left to wait for)."""
    waiting = {t.id: set(t.depends_on) for t in tasks}
    while waiting:
        ready = [tid for tid, deps in waiting.items() if not deps]
        if not ready:
            return True
        for tid in ready:
            del waiting[tid]
        for deps in waiting.values():
            deps.difference_update(ready)
    return False


def validate_plan(plan: QueryPlan, *, soft_cap: int, hard_cap: int) -> PlanStatus:
    """Check a plan and return the status it should have. Raises PlanError listing EVERY problem.

    Checks: ids unique, dependencies exist and do not loop, every task is scoped to an entity,
    no two tasks identical, count within the hard cap. Over the soft cap the plan is valid but
    needs the user's confirmation (cost estimate) before it runs."""
    problems: list[str] = []
    tasks = plan.tasks
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate task ids")
    for task in tasks:
        missing = [d for d in task.depends_on if d not in ids]
        if missing:
            problems.append(f"{task.id} depends on unknown task(s) {missing}")
        if task.id in task.depends_on:
            problems.append(f"{task.id} depends on itself")
        if task.entity is None:
            problems.append(f"{task.id} has no entity: every task must be scoped")
    if _cycle(tasks):
        problems.append("dependencies form a loop")
    signatures = [
        (t.entity.id if t.entity else None, t.period, t.metric, t.dimension, t.question)
        for t in tasks
    ]
    if len(set(map(repr, signatures))) != len(signatures):
        problems.append("two tasks are identical")
    if len(tasks) > hard_cap:
        problems.append(f"{len(tasks)} tasks is over the hard limit of {hard_cap}")
    if plan.status == "ready" and not tasks:
        problems.append("a plan that is ready has no tasks")
    if problems:
        raise PlanError(problems)
    if plan.status == "ready" and len(tasks) > soft_cap:
        return "needs_confirmation"
    return plan.status
