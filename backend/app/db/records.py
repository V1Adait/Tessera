"""Typed rows read from / written to the database (Pydantic, like every other boundary)."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant"]
MessageStatus = Literal["streaming", "complete", "stopped", "error"]
TraceKind = Literal[
    "chat", "summary", "classify", "extract", "plan", "embed", "agent", "intake"
]
CostSource = Literal["provider", "computed", "unknown"]
TraceStatus = Literal["ok", "error", "rejected", "cancelled"]
DocumentStatus = Literal[
    "uploaded", "parsing", "parsed", "needs_review", "needs_ocr", "failed"
]
DocumentKind = Literal["pdf", "docx", "xlsx", "pptx", "html", "csv", "txt", "md"]
BlockKind = Literal["heading", "paragraph", "list_item", "table", "figure"]


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
    ticker: str | None = None
    aliases: list[str] = Field(default_factory=list)
    domain: str | None = None  # the company's website, for its logo


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
    cost_source: CostSource | None = (
        None  # provider | computed | unknown; None = no paid call
    )
    latency_ms: int | None = None
    first_token_ms: int | None = None
    error_code: str | None = None
    created_at: datetime | None = None  # filled in by the database


IndexStatus = Literal["pending", "indexing", "indexed", "failed"]


class DocumentRecord(BaseModel):
    id: UUID
    workspace_id: UUID
    entity_id: UUID | None
    filename: str
    kind: DocumentKind
    sha256: str
    size_bytes: int
    status: DocumentStatus
    status_detail: str | None
    parser: str | None
    page_count: int | None
    char_count: int | None
    report: dict[str, Any]
    created_at: datetime
    parsed_at: datetime | None
    index_status: IndexStatus = "pending"  # are its chunks in the vector index?
    index_detail: str | None = None
    indexed_model: str | None = None
    indexed_at: datetime | None = None


class BlockRecord(BaseModel):
    """One parsed block, as stored. Chunking reads these in `seq` order."""

    id: UUID
    document_id: UUID
    seq: int
    type: BlockKind
    text: str
    level: int | None
    location: dict[str, Any]
    data: dict[str, Any] | None
    confidence: float
    flags: list[str]
    char_start: int
    char_end: int


ChunkKindName = Literal["prose", "table", "figure"]


class SectionRecord(BaseModel):
    """A parent: a heading and what is under it."""

    id: UUID
    document_id: UUID
    ordinal: int
    path: list[str]
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    token_count: int


class ChunkRecord(BaseModel):
    """A child chunk, as stored. `text` is what quotes are checked against; `search_text` is
    only for finding it."""

    id: UUID
    document_id: UUID
    section_id: UUID
    ordinal: int
    kind: ChunkKindName
    text: str
    search_text: str
    section_path: list[str]
    block_seqs: list[int]
    char_start: int
    char_end: int
    token_count: int
    page_start: int | None
    page_end: int | None
    sheet: str | None
    slide: int | None
    cell_range: str | None
    units: str | None
    confidence: float
    flags: list[str]
    content_hash: str


class ChunkHit(BaseModel):
    """A full-text search result: the chunk, its document, and the score."""

    chunk: ChunkRecord
    filename: str
    score: float
