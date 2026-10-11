"""Typed rows read from / written to the database (Pydantic, like every other boundary)."""

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

MessageRole = Literal["user", "assistant"]
MessageStatus = Literal["streaming", "complete", "stopped", "error"]
TraceKind = Literal["chat", "summary"]
TraceStatus = Literal["ok", "error", "rejected", "cancelled"]


class WorkspaceRecord(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime


class EntityRecord(BaseModel):
    """One row of the matrix: a company, vendor or case."""

    id: UUID
    workspace_id: UUID
    name: str
    position: int
    created_at: datetime


class ConversationRecord(BaseModel):
    id: UUID
    workspace_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class StoredMessage(BaseModel):
    id: UUID
    conversation_id: UUID
    seq: int
    role: MessageRole
    content: str
    status: MessageStatus
    tokens: int | None
    request_id: UUID | None
    created_at: datetime

    @property
    def usable_in_prompt(self) -> bool:
        """Only real answers go back to a model: not failed or empty rows."""
        return self.content != "" and self.status in ("complete", "stopped")


class StoredSummary(BaseModel):
    id: UUID
    seq: int
    conversation_id: UUID
    previous_summary_id: UUID | None
    summarised_through_message_id: UUID  # the pointer
    content: str
    tokens: int | None
    prompt_version: str
    model: str | None
    request_id: UUID | None
    created_at: datetime


class TrimEvent(BaseModel):
    id: UUID
    conversation_id: UUID
    request_id: UUID
    reason: Literal["budget"]
    messages_dropped: int
    tokens_dropped: int | None
    budget_tokens: int
    dropped_through_message_id: UUID | None
    created_at: datetime


class TraceRecord(BaseModel):
    """One handled request. Fields not known yet (or not applicable) stay None."""

    request_id: UUID
    kind: TraceKind
    status: TraceStatus
    conversation_id: UUID | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    first_token_ms: int | None = None
    error_code: str | None = None
    created_at: datetime | None = None  # filled in by the database
