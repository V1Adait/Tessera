"""What every parser produces: a ParsedDocument made of typed, located blocks.

This is the contract with chunking. Text is stored VERBATIM (after safe cleaning), because a
cited quote must later be found in it with a plain string match.
"""

from typing import Literal

from pydantic import BaseModel, Field

DocKind = Literal["pdf", "docx", "xlsx", "pptx", "html", "csv", "txt", "md"]
BlockType = Literal["heading", "paragraph", "list_item", "table", "figure"]
PageStatus = Literal["ok", "needs_ocr", "empty", "failed"]
DocStatus = Literal["parsed", "needs_review", "needs_ocr", "failed"]


class Location(BaseModel):
    """Where a block came from, in the terms a person would use to find it again."""

    page: int | None = None  # PDF page, 1-based
    page_end: int | None = None  # set when a table continues over several pages
    bbox: tuple[float, float, float, float] | None = (
        None  # x0, top, x1, bottom (PDF points)
    )
    sheet: str | None = None
    cell_range: str | None = None  # "A1:F45"
    slide: int | None = None
    section_path: list[str] = Field(default_factory=list)  # headings above this block
    index: int | None = None  # element order inside its container


class TableData(BaseModel):
    """A table as structured cells. Merged cells appear once, at their origin; the cells they
    cover are empty strings, so every row has the same number of columns."""

    rows: list[list[str]]
    header_rows: int = 0
    units: str | None = None  # e.g. "in millions", found in the text above the table
    has_merged_cells: bool = False


class Block(BaseModel):
    type: BlockType
    text: str  # verbatim text; for a table, its rows rendered as 'cell | cell' lines
    location: Location = Field(default_factory=Location)
    level: int | None = None  # heading level, 1 is the highest
    table: TableData | None = None
    labels: list[str] = Field(
        default_factory=list
    )  # figures: real text found inside the chart
    confidence: float = 1.0
    flags: list[str] = Field(default_factory=list)


class PageReport(BaseModel):
    """One PDF page, sheet or slide: did we get its content?"""

    index: int
    label: str | None = None
    status: PageStatus = "ok"
    chars: int = 0
    flags: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    kind: DocKind
    parser: str
    blocks: list[Block] = Field(default_factory=list)
    pages: list[PageReport] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)  # document-level warnings


class ParseError(Exception):
    """The file could not be parsed. `message` is safe to show to the user."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def render_table(rows: list[list[str]]) -> str:
    """A table as text: one line per row, cells separated by ' | '. This is the text that is
    searched and cited; the structured cells stay in TableData."""
    return "\n".join(" | ".join(row) for row in rows)


def assign_offsets(blocks: list[Block]) -> list[tuple[int, int]]:
    """Character offsets of each block in the document's canonical text, which is every block's
    text joined by a blank line. Returns [(start, end)] in block order."""
    offsets: list[tuple[int, int]] = []
    position = 0
    for block in blocks:
        offsets.append((position, position + len(block.text)))
        position += len(block.text) + 2  # the "\n\n" separator
    return offsets


def canonical_text(blocks: list[Block]) -> str:
    return "\n\n".join(block.text for block in blocks)
