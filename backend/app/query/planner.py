"""The LLM planner for open questions ("is Acme a good investment?").

    ask the model to pick rubric dimensions and word one factual question for each
    check its answer in code  (known dimensions, no repeats, sane questions, not too many)
    wrong?  tell it what was wrong and ask once more
    still wrong or the call fails?  use the rubric's own default questions, and SAY so in the notes

The model cannot add a dimension and cannot touch the entity: the entity comes from our own
slot matching, and tasks are built here, in code, from the checked answer.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.query.llm_json import ModelCallError, ask_json, neutralise
from backend.app.query.models import EntityRef, SubTask
from backend.app.query.prompts import PLANNER_PROMPT, PLANNER_PROMPT_VERSION
from backend.app.query.rubric import INVESTING_RUBRIC, dimension
from backend.app.tracing import Recorder, no_recorder

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 300
_TAGS = ("question", "company", "problem")


class PlannerOutputError(ValueError):
    """The model's plan failed our checks. The message is fed back to it on the retry."""


@dataclass
class PlannerResult:
    tasks: list[SubTask]
    notes: list[str] = field(default_factory=list)


def check_planner_output(
    data: dict[str, Any], max_dimensions: int
) -> list[tuple[str, str]]:
    """The (dimension id, question) pairs if the output is acceptable, else PlannerOutputError."""
    items = data.get("tasks")
    if not isinstance(items, list) or not items:
        raise PlannerOutputError('"tasks" must be a non-empty list')
    if len(items) > max_dimensions:
        raise PlannerOutputError(
            f"pick at most {max_dimensions} dimensions, not {len(items)}"
        )
    chosen: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise PlannerOutputError("each task must be an object")
        dim_id, question = item.get("dimension"), item.get("question")
        if not isinstance(dim_id, str) or dimension(dim_id) is None:
            raise PlannerOutputError(
                f"unknown dimension {dim_id!r}: use only the listed ids"
            )
        if any(dim_id == seen for seen, _ in chosen):
            raise PlannerOutputError(f"dimension {dim_id!r} appears twice")
        if not isinstance(question, str) or not (
            MIN_QUESTION_CHARS <= len(question.strip()) <= MAX_QUESTION_CHARS
        ):
            raise PlannerOutputError(
                f"the question for {dim_id!r} must be {MIN_QUESTION_CHARS}-{MAX_QUESTION_CHARS} characters"
            )
        chosen.append((dim_id, question.strip()))
    return chosen


def default_pairs() -> list[tuple[str, str]]:
    return [(d.id, d.default_question) for d in INVESTING_RUBRIC]


def build_tasks(
    pairs: list[tuple[str, str]], entities: list[EntityRef]
) -> list[SubTask]:
    """One judge task per (entity, chosen dimension). {entity} in a question is filled in."""
    tasks: list[SubTask] = []
    for entity in entities:
        for dim_id, text in pairs:
            tasks.append(
                SubTask(
                    id=f"t{len(tasks) + 1}",
                    kind="judge",
                    question=text.replace("{entity}", entity.name),
                    entity=entity,
                    dimension=dim_id,
                )
            )
    return tasks


def _system_prompt() -> str:
    menu = "\n".join(f"- {d.id}: {d.label}" for d in INVESTING_RUBRIC)
    return f"{PLANNER_PROMPT}\n\nDimensions you may choose from:\n{menu}"


class RubricPlanner:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        provider: str,
        model: str,
        max_dimensions: int,
        recorder: Recorder = no_recorder,
    ) -> None:
        self._llm = llm
        self._provider = provider
        self._model = model
        self._max = max_dimensions
        self._recorder = recorder

    async def plan(self, question: str, entities: list[EntityRef]) -> PlannerResult:
        names = ", ".join(e.name for e in entities)
        problem: str | None = None
        for attempt in (1, 2):
            try:
                data = await self._call(question, names, problem)
                pairs = check_planner_output(data, self._max)
                return PlannerResult(build_tasks(pairs, entities))
            except PlannerOutputError as exc:
                problem = str(exc)
                logger.warning("planner output rejected (attempt %s): %s", attempt, exc)
            except ModelCallError as exc:
                # bad JSON is worth one retry; a failed call (timeout, auth) is not
                if exc.code != "bad_output":
                    return self._fallback(entities, f"planner call failed ({exc.code})")
                problem = "the reply was not valid JSON"
        return self._fallback(entities, f"planner output rejected twice ({problem})")

    async def _call(
        self, question: str, names: str, problem: str | None
    ) -> dict[str, Any]:
        body = (
            f"<question>{neutralise(question, _TAGS)}</question>\n"
            f"<company>{neutralise(names, _TAGS)}</company>"
        )
        if problem:
            body += f"\n<problem>Your last answer was rejected: {problem}. Fix it.</problem>"
        return await ask_json(
            self._llm,
            [SystemMessage(content=_system_prompt()), HumanMessage(content=body)],
            recorder=self._recorder,
            kind="plan",
            provider=self._provider,
            model=self._model,
            prompt_version=PLANNER_PROMPT_VERSION,
        )

    @staticmethod
    def _fallback(entities: list[EntityRef], why: str) -> PlannerResult:
        return PlannerResult(
            build_tasks(default_pairs(), entities),
            [f"planner fell back to the full rubric: {why}"],
        )
