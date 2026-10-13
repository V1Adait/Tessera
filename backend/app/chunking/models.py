"""The shapes chunking works with.

    blocks (from the parser)  ->  build_chunks  ->  ChunkSet
                                                      parents  one per section (the thing the model READS)
                                                      chunks   small pieces inside a parent (the thing we SEARCH and CITE)

Two strings per chunk, never mixed up:
    text         the exact words from the document. Quotes are checked against this and only this.
    search_text  a short context line (document, section path, units) + text. Only for finding.
"""

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

ChunkKind = Literal["prose", "table", "figure"]


class DocContext(BaseModel):
    """What chunking knows about the document as a whole, for the search context line."""

    label: str  # the title, or the file name when there is no title
    entity: str | None = None  # the company this document belongs to, if assigned


class ChunkLimits(BaseModel):
    """Sizes in estimated tokens (backend.app.context.estimate_tokens: 3 characters each,
    deliberately high). `target` is where merging stops; `max` is the hard ceiling."""

    target_tokens: int = Field(default=512, gt=0)
    max_tokens: int = Field(default=800, gt=0)
    parent_max_tokens: int = Field(default=3000, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> ChunkLimits:
        if not self.target_tokens <= self.max_tokens <= self.parent_max_tokens:
            raise ValueError("need target_tokens <= max_tokens <= parent_max_tokens")
        return self


class Section(BaseModel):
    """The PARENT: the blocks under one heading (or a window of them if the section is huge).

    `path` is the headings above and including this section's own, outermost first. A section
    never contains a deeper section: every heading starts a new one."""

    ordinal: int  # position in the document, from 0
    path: list[str]
    char_start: (
        int  # span in the canonical text, from the heading through the last child
    )
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0


class Chunk(BaseModel):
    """The CHILD: what gets searched, embedded and cited."""

    ordinal: int  # position in the document, from 0
    parent_ordinal: int
    kind: ChunkKind
    text: str
    search_text: str
    section_path: list[str]
    block_seqs: list[int]  # indexes into the input block list, ascending
    char_start: (
        int  # span of the ORIGINAL text this chunk covers (canonical text offsets)
    )
    char_end: int
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    slide: int | None = None
    cell_range: str | None = None
    units: str | None = None  # tables: "in millions" and the like
    confidence: float = 1.0  # the lowest confidence of the blocks it came from
    flags: list[str] = Field(default_factory=list)  # every flag of those blocks, sorted

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Identifies what would be embedded. Same search_text -> same hash -> no new embedding."""
        return hashlib.sha256(self.search_text.encode()).hexdigest()


class ChunkSet(BaseModel):
    parents: list[Section] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
