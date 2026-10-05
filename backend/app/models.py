from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import Literal


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
    usage: Usage
    text: str


StreamEvent = Delta | Done
