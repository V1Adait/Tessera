from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.intake.models import TICKER, normalise_domain


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    system: str | None = None
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None


class ChatInput(BaseModel):
    """What a client sends to POST /chat: ONE new message. The server owns the history, so the
    client cannot forge earlier turns or send system messages."""

    conversation_id: UUID | None = None  # None starts a new conversation
    # Which workspace a NEW conversation belongs to (None: the default workspace). An existing
    # conversation already has one, so sending both is rejected as ambiguous.
    workspace_id: UUID | None = None
    message: str = Field(min_length=1, max_length=20_000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @model_validator(mode="after")
    def one_of_conversation_or_workspace(self) -> ChatInput:
        if self.conversation_id is not None and self.workspace_id is not None:
            raise ValueError("send conversation_id or workspace_id, not both")
        return self


def _clean_name(value: str) -> str:
    """Names are stored trimmed, with no control characters and no blanks."""
    value = value.strip()
    if not value:
        raise ValueError("name must not be blank")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("name must not contain control characters")
    return value


class NewWorkspace(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def clean(cls, value: str) -> str:
        return _clean_name(value)


class NewEntity(BaseModel):
    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def clean(cls, value: str) -> str:
        return _clean_name(value)


class EntityUpdate(BaseModel):
    """Body of PATCH /workspaces/{id}/entities/{entity_id}: a company's ticker and/or website.

    A field that is left out stays as it is; a field sent as null (or blank) is cleared."""

    ticker: str | None = None
    domain: str | None = None

    @field_validator("ticker")
    @classmethod
    def clean_ticker(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        cleaned = value.strip().upper()
        if not TICKER.match(cleaned):
            raise ValueError("a ticker is letters and digits, like PLTR")
        return cleaned

    @field_validator("domain")
    @classmethod
    def clean_domain(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        cleaned = normalise_domain(value)
        if cleaned is None:
            raise ValueError(
                "that does not look like a website domain, like palantir.com"
            )
        return cleaned

    @model_validator(mode="after")
    def something_to_change(self) -> EntityUpdate:
        if not self.model_fields_set:
            raise ValueError("send a ticker or a domain to change")
        return self


class Usage(BaseModel):
    tokens_in: int
    tokens_out: int


class Delta(BaseModel):
    type: Literal["delta"] = "delta"
    text: str


class Done(BaseModel):
    type: Literal["done"] = "done"
    request_id: UUID
    usage: Usage
    # Things the person should know about this answer (an unverifiable citation, a step limit).
    warnings: list[str] = Field(default_factory=list)


class ToolEvent(BaseModel):
    """The agent used a tool. Sent when it starts (status 'running') and again when it ends, with
    the same `id`, so the screen can show a step and then tick it off."""

    type: Literal["tool"] = "tool"
    id: str
    name: str
    label: str  # plain words: "Searching the documents"
    status: Literal["running", "ok", "error", "needs_confirmation"]
    summary: str = ""
    changed_matrix: bool = False  # the grid should reload


class ConfirmationEvent(BaseModel):
    """The agent wants to do something costly or irreversible. Nothing has happened yet: a person
    confirms or cancels via POST /actions/{id}/confirm or /cancel."""

    type: Literal["confirmation"] = "confirmation"
    action_id: UUID
    tool: str
    risk: Literal["costly", "destructive"]
    summary: str
    # What the card shows beyond the summary; for a column plan, the columns and the cost.
    plan: dict[str, Any] | None = None


ErrorCode = Literal[
    "auth",  # bad or missing provider key
    "model_not_found",  # invalid model name
    "rate_limited",
    "timeout",
    "unavailable",  # network failure or provider 5xx
    "bad_request",
    "context_too_large",  # the prompt cannot fit the window even after trimming history
    "budget",  # a cost, step or time limit was reached
    "unknown",
]


class ErrorEvent(BaseModel):
    """Terminal event: the stream ended in failure. Safe to show to a client."""

    type: Literal["error"] = "error"
    request_id: UUID
    code: ErrorCode
    message: str
    retryable: bool


StreamEvent = Delta | Done | ErrorEvent | ToolEvent | ConfirmationEvent
