from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
