"""Data shapes for query handling: what a classifier says, what we found in the question, and
the plan (the list of sub-tasks) that comes out. All of them are Pydantic models, so a bad
shape is rejected where it is built, not three steps later."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.query.classes import QuestionType

# --- classification ---------------------------------------------------------------------


class Prediction(BaseModel):
    """One classifier's answer."""

    label: QuestionType
    confidence: float = Field(ge=0.0, le=1.0)
    needs_decomposition: bool | None = None
    classifier: str  # which implementation answered (goes in the trace)


class Routing(BaseModel):
    """What the router decided, and how it got there."""

    question_type: QuestionType  # the type the rest of the pipeline uses
    prediction: (
        Prediction | None
    )  # the primary classifier's answer (None if all failed)
    second_opinion: Prediction | None = None
    low_confidence: bool = False  # the gate fired
    fell_back: bool = False  # the primary failed, a backup answered
    reason: str  # one plain sentence, for logs and the UI


# --- what the question mentions ----------------------------------------------------------


class Period(BaseModel):
    """A year or a quarter named in the question. `label` is how we show it ('FY2024')."""

    kind: Literal["year", "quarter"]
    year: int
    quarter: int | None = Field(default=None, ge=1, le=4)

    @property
    def label(self) -> str:
        return f"Q{self.quarter} {self.year}" if self.quarter else f"FY{self.year}"


class EntityRef(BaseModel):
    id: UUID
    name: str


class Slots(BaseModel):
    entities: list[EntityRef] = Field(default_factory=list)
    periods: list[Period] = Field(default_factory=list)
    metrics: list[str] = Field(
        default_factory=list
    )  # phrases that appear in the question


# --- the plan ---------------------------------------------------------------------------

TaskKind = Literal["lookup", "judge"]
# lookup: one fact the documents state. judge: an open question answered from several passages.

Combine = Literal["none", "compare", "trend", "filter", "assess"]
# What the final step does with the finished cells. The arithmetic is done in code later; the
# plan only names it.

PlanStatus = Literal[
    "ready",  # run it
    "needs_clarification",  # a slot is missing or unknown: ask the user, do not guess
    "needs_confirmation",  # valid but large: show a cost estimate first
    "no_retrieval",  # nothing to look up (chat)
]


class SubTask(BaseModel):
    id: str
    kind: TaskKind
    question: str  # what to find out, in words the retriever can use
    entity: EntityRef | None = None
    period: Period | None = None  # None means "whatever the documents cover"
    metric: str | None = None
    dimension: str | None = None  # the rubric dimension, for evaluative tasks
    depends_on: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    question: str
    question_type: QuestionType
    status: PlanStatus
    combine: Combine = "none"
    tasks: list[SubTask] = Field(default_factory=list)
    clarification: str | None = None  # set when status is needs_clarification
    notes: list[str] = Field(
        default_factory=list
    )  # fallbacks and assumptions, never silent
    routing: Routing | None = None
    slots: Slots = Field(default_factory=Slots)


class PlanRequest(BaseModel):
    """Body of POST /workspaces/{id}/query/plan."""

    question: str = Field(min_length=1, max_length=2000)
