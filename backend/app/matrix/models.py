"""Data shapes for the matrix: column specs, cells, evidence, and what the model may say.

Two kinds of shape, kept apart on purpose:

    ModelReply        what the language model SAYS. Unchecked claims: an answer, a status and
                      quotes it says support it. It never reaches the database as it is.
    CellOutcome       what the code DECIDED after checking: quotes found in the cited text,
                      values parsed, the status set. Only this is stored.
"""

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

ColumnType = Literal["text", "number", "date", "currency", "select", "yes_no", "status"]
CellStatus = Literal[
    "answered",
    "not_applicable",
    "not_found",
    "conflicting",
    "low_confidence",
    "pending_or_error",
]

MAX_ALLOWED_VALUES = 50
MAX_EVIDENCE = 5


# --- columns ----------------------------------------------------------------------------------


class ColumnDraft(BaseModel):
    """What a person gives to define (or change) a column."""

    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=2000)
    type: ColumnType = "text"
    allowed_values: list[str] = Field(
        default_factory=list, max_length=MAX_ALLOWED_VALUES
    )
    applicability: str | None = Field(default=None, max_length=1000)
    evidence_required: bool = True
    depends_on: list[UUID] = Field(default_factory=list)

    @field_validator("name", "prompt")
    @classmethod
    def _trimmed(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("applicability")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None

    @field_validator("allowed_values")
    @classmethod
    def _clean_values(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v.strip()]
        seen: set[str] = set()
        for value in cleaned:
            if value.casefold() in seen:
                raise ValueError(f"allowed value repeated: {value!r}")
            seen.add(value.casefold())
        if any(len(v) > 80 for v in cleaned):
            raise ValueError("an allowed value is longer than 80 characters")
        return cleaned

    @model_validator(mode="after")
    def _type_rules(self) -> ColumnDraft:
        if self.type == "select" and len(self.allowed_values) < 2:
            raise ValueError("a select column needs at least two allowed values")
        if self.type not in ("select", "status") and self.allowed_values:
            raise ValueError("allowed values only apply to select and status columns")
        return self


class ColumnPatch(BaseModel):
    """A change to a column. Any field left out keeps its current value; a change makes a new
    version, and the cells that answered the old one are marked stale."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    prompt: str | None = Field(default=None, min_length=1, max_length=2000)
    type: ColumnType | None = None
    allowed_values: list[str] | None = None
    applicability: str | None = None
    clear_applicability: bool = False
    evidence_required: bool | None = None
    depends_on: list[UUID] | None = None


class ColumnRecord(BaseModel):
    """A column as stored, at its current version."""

    id: UUID
    workspace_id: UUID
    position: int
    version: int
    name: str
    prompt: str
    type: ColumnType
    allowed_values: list[str]
    applicability: str | None
    evidence_required: bool
    depends_on: list[UUID]
    created_at: datetime
    updated_at: datetime


# --- what the model may say -------------------------------------------------------------------


class ProposedEvidence(BaseModel):
    chunk: str = Field(
        min_length=1, max_length=20
    )  # a label such as "C3", never a database id
    quote: str = Field(min_length=1, max_length=2000)


class ModelReply(BaseModel):
    """The model's reply, validated for shape only. Whether it is TRUE is checked in code."""

    status: Literal["answered", "not_found", "not_applicable", "conflicting"]
    answer: str | None = Field(default=None, max_length=2000)
    reason: str | None = Field(default=None, max_length=500)
    evidence: list[ProposedEvidence] = Field(
        default_factory=list, max_length=MAX_EVIDENCE
    )

    @model_validator(mode="after")
    def _consistent(self) -> ModelReply:
        if self.status == "answered":
            if not (self.answer and self.answer.strip()):
                raise ValueError("status is answered but there is no answer")
        elif self.status == "conflicting":
            if len(self.evidence) < 2:
                raise ValueError("conflicting needs at least two pieces of evidence")
        elif not (self.reason and self.reason.strip()):
            raise ValueError(f"status {self.status} needs a reason")
        return self


# --- what the code decided --------------------------------------------------------------------


class VerifiedEvidence(BaseModel):
    """A quote that was found, character for character, in the chunk it cites."""

    document_id: UUID
    chunk_id: (
        UUID | None
    )  # None only if the chunk is gone; the quote and its range still stand
    quote: str  # copied from the chunk text, so it is verbatim even if the model's spacing differed
    char_start: int
    char_end: int
    exact_range: bool
    page_start: int | None = None
    page_end: int | None = None
    chunk_confidence: float = 1.0
    chunk_flags: list[str] = Field(default_factory=list)


class CellOutcome(BaseModel):
    status: CellStatus
    detail: str | None = None
    value: Any = None  # normalised: see values.py for the shape per column type
    display: str | None = None
    value_type: ColumnType | None = None
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    model: str | None = None
    prompt_version: str | None = None


# --- stored cells -----------------------------------------------------------------------------


class EvidenceRecord(BaseModel):
    id: UUID
    cell_id: UUID
    seq: int
    document_id: UUID
    chunk_id: UUID | None
    page_start: int | None
    page_end: int | None
    char_start: int
    char_end: int
    exact_range: bool
    quote: str


class CellRecord(BaseModel):
    id: UUID
    workspace_id: UUID
    entity_id: UUID
    column_id: UUID
    column_version: int
    status: CellStatus
    status_detail: str | None
    value: Any
    display_value: str | None
    as_of: date | None
    derivation: dict[str, Any] | None
    stale: bool
    pinned: bool
    model: str | None
    prompt_version: str | None
    created_at: datetime
    updated_at: datetime
    evidence: list[EvidenceRecord] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    """Body of POST /workspaces/{id}/cells/extract."""

    entity_id: UUID
    column_id: UUID
    replace_pinned: bool = (
        False  # overwriting a cell a person set needs this to be true
    )
