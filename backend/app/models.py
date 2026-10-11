from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


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


ErrorCode = Literal[
    "auth",  # bad or missing provider key
    "model_not_found",  # invalid model name
    "rate_limited",
    "timeout",
    "unavailable",  # network failure or provider 5xx
    "bad_request",
    "context_too_large",  # the prompt cannot fit the window even after trimming history
    "unknown",
]


class ErrorEvent(BaseModel):
    """Terminal event: the stream ended in failure. Safe to show to a client."""

    type: Literal["error"] = "error"
    request_id: UUID
    code: ErrorCode
    message: str
    retryable: bool


StreamEvent = Delta | Done | ErrorEvent
