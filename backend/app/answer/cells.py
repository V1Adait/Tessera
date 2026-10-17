"""The filled cells, as the text the model answers from.

After a plan is confirmed and filled, the chat answers the user's question FROM THE CELLS, not from
a fresh search and never from memory. This module turns the cells into a compact table of data:

    [Palantir · Revenue] answered: $1.0 billion
        quote: "total revenue of $1.0 billion" (q2.pdf)
    [Palantir · Total debt] not_found: the documents do not say

Every line starts with the label the model must copy to cite it. Cell text comes from documents, so
it is untrusted data: fenced in its own tag, and unable to close that tag.
"""

from collections.abc import Sequence
from uuid import UUID

from backend.app.db.records import EntityRecord
from backend.app.matrix.models import CellRecord, ColumnRecord
from backend.app.query.llm_json import neutralise

DATA_TAGS = ("cells", "question", "documents")
QUOTE_CHARS = 240
STATUS_WORDS = {
    "answered": "answered",
    "not_found": "not found",
    "not_applicable": "does not apply",
    "conflicting": "conflicting",
    "low_confidence": "low confidence",
    "pending_or_error": "could not be read",
}


def label(row: str, column: str) -> str:
    """How a cell is cited: [Palantir · Revenue]. Also what the model is told to copy."""
    return f"[{_one_line(row)} · {_one_line(column)}]"


def _one_line(text: str) -> str:
    return neutralise(
        " ".join(text.replace("[", "(").replace("]", ")").split()), DATA_TAGS
    )


def _cell_lines(
    row: EntityRecord,
    column: ColumnRecord,
    cell: CellRecord | None,
    filenames: dict[UUID, str],
    with_quote: bool,
) -> str:
    tag = label(row.name, column.name)
    if cell is None:
        return f"{tag} not filled: no answer was produced for this cell"
    word = STATUS_WORDS.get(cell.status, cell.status)
    if cell.status == "answered":
        head = f"{tag} {word}: {_one_line(cell.display_value or '(no value)')}"
        if cell.stale:
            head += " (out of date: the question changed after this was filled)"
    else:
        reason = _one_line(cell.status_detail) if cell.status_detail else ""
        shown = (
            f" (the model proposed: {_one_line(cell.display_value)})"
            if cell.display_value
            else ""
        )
        head = f"{tag} {word}" + (f": {reason}" if reason else "") + shown
    if with_quote and cell.evidence:
        ev = cell.evidence[0]
        quote = _one_line(ev.quote)
        if len(quote) > QUOTE_CHARS:
            quote = quote[: QUOTE_CHARS - 1].rstrip() + "…"
        source = filenames.get(ev.document_id, "document")
        head += f'\n    quote: "{quote}" ({_one_line(source)})'
    return head


def build_cell_table(
    rows: Sequence[EntityRecord],
    columns: Sequence[ColumnRecord],
    cells: Sequence[CellRecord],
    filenames: dict[UUID, str],
    *,
    max_chars: int = 24_000,
) -> str:
    """The table, row by row. If it would be too long, quotes go first, then whole cells, and the
    text says how many were left out: the model must never think a cut table is the whole table."""
    by_key = {(c.entity_id, c.column_id): c for c in cells}

    def render(with_quote: bool, limit: int | None) -> tuple[str, int]:
        blocks: list[str] = []
        shown = 0
        size = 0
        total = len(rows) * len(columns)
        for row in rows:
            lines = [f"Row: {_one_line(row.name)}"]
            for column in columns:
                line = _cell_lines(
                    row, column, by_key.get((row.id, column.id)), filenames, with_quote
                )
                if limit is not None and size + len(line) > limit:
                    return "\n".join(blocks + ["\n".join(lines)]), total - shown
                lines.append(line)
                size += len(line) + 1
                shown += 1
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks), 0

    text, missing = render(True, None)
    if len(text) > max_chars:
        text, missing = render(False, None)
    if len(text) > max_chars:
        text, missing = render(False, max_chars)
    note = (
        f"\n\n(The table was cut: {missing} cell(s) are not shown. Say so if it matters.)"
        if missing
        else ""
    )
    return f"<cells>\n{text}{note}\n</cells>"
