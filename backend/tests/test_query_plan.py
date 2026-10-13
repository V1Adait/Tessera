"""Plan templates (one per class) and validate_plan, the safety net over every plan."""

from uuid import uuid4

import pytest

from backend.app.query.models import EntityRef, Period, QueryPlan, Slots, SubTask
from backend.app.query.plan import PlanError, expand, validate_plan


def ref(name: str) -> EntityRef:
    return EntityRef(id=uuid4(), name=name)


def year(y: int) -> Period:
    return Period(kind="year", year=y)


ACME, BETA, GAMMA = ref("Acme"), ref("Beta"), ref("Gamma")
WORKSPACE = [ACME, BETA, GAMMA]


def run(kind: str, slots: Slots, question: str = "q"):
    return expand(kind, question, slots, WORKSPACE)


# --- lookup -----------------------------------------------------------------------------


def test_lookup_is_one_task_that_keeps_the_users_question():
    result = run("lookup", Slots(entities=[ACME]), "What was Acme's FY24 revenue?")

    assert [t.question for t in result.tasks] == ["What was Acme's FY24 revenue?"]
    assert result.combine == "none"
    assert result.clarification is None


def test_lookup_without_a_company_asks_instead_of_guessing():
    result = run("lookup", Slots())

    assert result.tasks == []
    assert result.clarification


def test_lookup_naming_two_companies_splits_and_says_so():
    result = run("lookup", Slots(entities=[ACME, BETA]))

    assert len(result.tasks) == 2
    assert result.combine == "compare"
    assert result.notes


# --- multi_field ------------------------------------------------------------------------


def test_multi_field_makes_one_task_per_metric():
    slots = Slots(entities=[ACME], metrics=["revenue", "ebitda", "net debt"])

    result = run("multi_field", slots)

    assert [t.metric for t in result.tasks] == ["revenue", "ebitda", "net debt"]
    assert {t.entity.name for t in result.tasks if t.entity} == {"Acme"}
    assert result.tasks[0].question == "revenue for Acme"


def test_multi_field_includes_the_period_in_each_task():
    slots = Slots(entities=[ACME], metrics=["revenue"], periods=[year(2024)])

    assert run("multi_field", slots).tasks[0].question == "revenue for Acme in FY2024"


def test_multi_field_without_named_facts_asks_which():
    assert run("multi_field", Slots(entities=[ACME])).clarification


# --- comparison -------------------------------------------------------------------------


def test_comparison_makes_one_task_per_company():
    slots = Slots(entities=[ACME, BETA], metrics=["operating margin"])

    result = run("comparison", slots)

    assert [t.entity.name for t in result.tasks if t.entity] == ["Acme", "Beta"]
    assert result.combine == "compare"


def test_comparison_with_an_unknown_second_company_asks_which():
    result = run("comparison", Slots(entities=[ACME]), "Acme vs Zed on margin")

    assert result.tasks == []
    assert result.clarification and "Acme" in result.clarification


def test_comparison_inside_one_company_stays_one_task_and_says_so():
    result = run(
        "comparison", Slots(entities=[ACME]), "Which of Acme's segments grew fastest?"
    )

    assert len(result.tasks) == 1
    assert result.combine == "compare"
    assert result.notes


# --- trend ------------------------------------------------------------------------------


def test_trend_makes_one_task_per_period():
    slots = Slots(entities=[ACME], periods=[year(2022), year(2023), year(2024)])

    result = run("trend", slots)

    assert [t.period.year for t in result.tasks if t.period] == [2022, 2023, 2024]
    assert result.combine == "trend"
    assert result.notes == []


def test_trend_without_a_period_range_reads_the_whole_series_and_says_so():
    result = run("trend", Slots(entities=[ACME]))

    assert len(result.tasks) == 1 and result.tasks[0].period is None
    assert result.combine == "trend"
    assert result.notes


# --- aggregation ------------------------------------------------------------------------


def test_aggregation_with_no_company_named_runs_over_the_whole_workspace():
    result = run("aggregation", Slots())

    assert len(result.tasks) == 3
    assert result.combine == "filter"
    assert result.notes


def test_aggregation_in_an_empty_workspace_asks():
    result = expand("aggregation", "q", Slots(), [])

    assert result.clarification


def test_unknown_class_has_no_template():
    with pytest.raises(ValueError):
        run("chat", Slots())


# --- validate_plan ----------------------------------------------------------------------


def task(task_id: str, *, deps: list[str] | None = None, entity=ACME, q=None):
    return SubTask(
        id=task_id,
        kind="lookup",
        question=q or f"question {task_id}",
        entity=entity,
        depends_on=deps or [],
    )


def plan_of(tasks: list[SubTask], status="ready") -> QueryPlan:
    return QueryPlan(question="q", question_type="lookup", status=status, tasks=tasks)


def check(tasks: list[SubTask], soft: int = 8, hard: int = 50, status="ready"):
    return validate_plan(plan_of(tasks, status), soft_cap=soft, hard_cap=hard)


def test_a_good_plan_stays_ready():
    assert check([task("a"), task("b", deps=["a"])]) == "ready"


def test_more_tasks_than_the_soft_cap_need_confirmation():
    tasks = [task(f"t{i}") for i in range(9)]

    assert check(tasks, soft=8) == "needs_confirmation"
    assert check(tasks[:8], soft=8) == "ready"


def test_more_tasks_than_the_hard_cap_are_refused():
    with pytest.raises(PlanError, match="hard limit"):
        check([task(f"t{i}") for i in range(6)], hard=5)


def test_duplicate_ids_are_refused():
    with pytest.raises(PlanError, match="duplicate task ids"):
        check([task("a"), task("a", q="other")])


def test_a_dependency_on_a_missing_task_is_refused():
    with pytest.raises(PlanError, match="unknown"):
        check([task("a", deps=["ghost"])])


def test_a_task_depending_on_itself_is_refused():
    with pytest.raises(PlanError, match="itself"):
        check([task("a", deps=["a"])])


def test_a_dependency_loop_is_refused():
    with pytest.raises(PlanError, match="loop"):
        check([task("a", deps=["c"]), task("b", deps=["a"]), task("c", deps=["b"])])


def test_a_diamond_of_dependencies_is_not_a_loop():
    tasks = [
        task("a"),
        task("b", deps=["a"]),
        task("c", deps=["a"]),
        task("d", deps=["b", "c"]),
    ]

    assert check(tasks) == "ready"


def test_a_task_without_an_entity_is_refused():
    with pytest.raises(PlanError, match="scoped"):
        check([task("a", entity=None)])


def test_identical_tasks_are_refused():
    with pytest.raises(PlanError, match="identical"):
        check([task("a", q="same"), task("b", q="same")])


def test_a_ready_plan_with_no_tasks_is_refused():
    with pytest.raises(PlanError, match="no tasks"):
        check([])


def test_every_problem_is_reported_not_just_the_first():
    with pytest.raises(PlanError) as caught:
        check([task("a", deps=["a"], entity=None), task("a", q="x")])

    assert len(caught.value.reasons) >= 3


def test_a_plan_that_needs_clarification_may_have_no_tasks():
    assert check([], status="needs_clarification") == "needs_clarification"
