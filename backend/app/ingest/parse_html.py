"""HTML parser, built for SEC filings: real <table> tags, headings, lists, and inline XBRL.

Content a person cannot see is NOT parsed (display:none, hidden attributes, XBRL header blocks),
because hidden text is the classic way to plant instructions for a model. It is counted and
reported instead.
"""

import re
from pathlib import Path

from bs4 import (
    BeautifulSoup,
    CData,
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
    Tag,
)

from backend.app.config import Settings
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import PageReport, ParsedDocument, ParseError
from backend.app.ingest.parse_text import decode_text

_SKIP = {
    "script",
    "style",
    "noscript",
    "template",
    "head",
    "svg",
    "iframe",
    "object",
    "embed",
    "canvas",
}
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_BLOCKS = {
    "p", "div", "li", "ul", "ol", "table", "pre", "blockquote", "section", "article", "header",
    "footer", "main", "aside", "nav", "figure", "figcaption", "form", "fieldset", "address",
    "dl", "dt", "dd", *_HEADINGS,
}  # fmt: skip
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em|%)?\s*(?:;|$)|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)",
    re.IGNORECASE,
)
_HIDDEN_TAGS = {
    "ix:header",
    "ix:hidden",
}  # inline XBRL metadata, never shown to a reader
_MAX_SPAN = 100
_MAX_CELLS = 200_000
_WS = re.compile(r"\s+")

# A heading written as styled text instead of an <h1>-<h6> tag (every SEC filing does this):
# a short line that is bold from end to end, or one that opens with "PART II" / "Item 1A.".
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(?:bold|bolder|[6-9]00)\b", re.IGNORECASE)
_BOLD_TAGS = {"b", "strong", "th"}
# "PART II" / "Item 1A. Risk Factors": the keyword, the number, then nothing or a title that starts
# with a capital. "Item 1A of this report discusses..." is a sentence, not a heading.
_PART = re.compile(r"^(?i:part)\s+[IVX]+(?:[.:]?\s*$|[.:]\s*[A-Z\[(]|\s+[A-Z\[(])")
_ITEM = re.compile(
    r"^(?i:item)\s+\d+[A-Ca-c]?(?:[.:]?\s*$|[.:]\s*[A-Z\[(0-9]|\s+[A-Z\[(])"
)
_MAX_HEADING_WORDS = 15
_MAX_HEADING_CHARS = 120


def _is_hidden(tag: Tag) -> bool:
    if tag.name in _HIDDEN_TAGS or tag.has_attr("hidden"):
        return True
    style = tag.get("style")
    return isinstance(style, str) and bool(_HIDDEN_STYLE.search(style))


def _collect(tag: Tag, out: list[str], hidden: list[int]) -> None:
    for child in tag.children:
        if isinstance(child, NavigableString):
            if not isinstance(
                child, (Comment, Doctype, CData, ProcessingInstruction, Declaration)
            ):
                out.append(str(child))
        elif isinstance(child, Tag):
            if _is_hidden(child):
                hidden[0] += 1  # text a reader cannot see: dropped, and counted
            elif child.name.lower() in _SKIP:
                continue
            elif child.name.lower() == "br":
                out.append(" ")
            else:
                _collect(child, out, hidden)


def _text_of(tag: Tag, hidden: list[int] | None = None) -> str:
    """The text a reader would SEE inside a tag: hidden descendants and scripts are skipped,
    <br> counts as a space. `hidden` collects how many hidden elements were dropped."""
    out: list[str] = []
    _collect(tag, out, hidden if hidden is not None else [0])
    return _WS.sub(" ", "".join(out)).strip()


def _is_bold(tag: Tag) -> bool:
    if tag.name.lower() in _BOLD_TAGS:
        return True
    style = tag.get("style")
    return isinstance(style, str) and bool(_BOLD_STYLE.search(style))


def _all_bold(block: Tag) -> bool:
    """True if every piece of visible text in `block` sits inside something bold."""
    seen = False
    for node in block.descendants:
        if not isinstance(node, NavigableString) or isinstance(
            node, (Comment, Doctype, CData, ProcessingInstruction, Declaration)
        ):
            continue
        if not str(node).strip():
            continue
        seen = True
        parent = node.parent
        while isinstance(parent, Tag):
            if _is_bold(parent):
                break
            if parent is block:
                return False
            parent = parent.parent
        else:
            return False
    return seen


def heading_level(block: Tag, text: str) -> int | None:
    """The heading level of a styled block of text, or None if it reads as an ordinary paragraph.

    PART I / PART II is level 1 and 'Item 1A.' level 2, whatever the styling; any other short
    line that is bold from end to end is level 3. Bold sentences (they end in punctuation),
    long bold passages, bare numbers (page numbers) and lowercase fragments stay paragraphs."""
    if not text or len(text) > _MAX_HEADING_CHARS:
        return None
    if len(text.split()) > _MAX_HEADING_WORDS:
        return None
    if text.endswith(".") and len(text.split()) > 6:
        return None  # a sentence, whatever it starts with
    if _PART.match(text):
        return 1
    if _ITEM.match(text):
        return 2
    if not _all_bold(block):
        return None
    if text.endswith((".", ",", ";", ":")) or not any(ch.isalpha() for ch in text):
        return None
    if text[0].islower():
        return None
    return 3


def _span(cell: Tag, attribute: str) -> int:
    try:
        return max(1, min(int(str(cell.get(attribute, "1")).strip() or "1"), _MAX_SPAN))
    except ValueError:
        return 1


def table_rows(
    table: Tag, hidden: list[int] | None = None
) -> tuple[list[list[str]], bool]:
    """A table's grid with row and column spans resolved. Spanned cells are empty strings and
    the text sits at the origin. Returns (rows, any_merged_cells)."""
    grid: dict[tuple[int, int], str] = {}
    merged = False
    row_count = 0
    width = 0
    trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    for r, tr in enumerate(trs):
        column = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, column) in grid:
                column += 1
            if len(grid) > _MAX_CELLS:
                raise ParseError(
                    "table_too_large", "A table in the document is too large to read."
                )
            colspan, rowspan = _span(cell, "colspan"), _span(cell, "rowspan")
            merged = merged or colspan > 1 or rowspan > 1
            grid[(r, column)] = _text_of(cell, hidden) if not _is_hidden(cell) else ""
            for dr in range(rowspan):
                for dc in range(colspan):
                    if (dr, dc) != (0, 0):
                        grid.setdefault((r + dr, column + dc), "")
            column += colspan
        row_count = max(row_count, r + 1)
        width = max(width, column)
    for r, c in grid:
        row_count = max(row_count, r + 1)
        width = max(width, c + 1)
    return [
        [grid.get((r, c), "") for c in range(width)] for r in range(row_count)
    ], merged


def clean_filing_table(rows: list[list[str]]) -> list[list[str]]:
    """SEC tables put '$', '%' and ')' in cells of their own. Rejoin them to their figures so
    '$' | '307,394' becomes '$307,394' and '(1,234' | ')' becomes '(1,234)'. The empty spacer
    columns this leaves are dropped by the table builder."""
    out: list[list[str]] = []
    for row in rows:
        cells = list(row)
        for i, cell in enumerate(cells):
            if cell in {")", "%", ")%", "%)"} and i > 0:
                j = i - 1
                while j >= 0 and not cells[j]:
                    j -= 1
                if j >= 0:
                    cells[j] += cell
                    cells[i] = ""
        for i, cell in enumerate(cells):
            if cell in {"$", "€", "£"}:
                j = i + 1
                while j < len(cells) and not cells[j]:
                    j += 1
                if j < len(cells):
                    # The merged figure goes in the column of the sign, because that is where
                    # rows WITHOUT a sign keep their figure; this keeps the whole column aligned.
                    cells[i] = cell + cells[j]
                    cells[j] = ""
        out.append(cells)
    return out


class _Walker:
    def __init__(self, builder: BlockBuilder) -> None:
        self.builder = builder
        self.hidden = [0]
        self.counter = 0

    def next_index(self) -> int:
        self.counter += 1
        return self.counter

    def units_context(self, table: Tag, rows: list[list[str]]) -> str:
        """The text that may say "(in millions)", NEAREST FIRST (the first match wins): the table's
        own header rows, its caption, then the end of the paragraph just above it. SEC tables put
        the units in the second header row ("Amount(in millions)") as often as in the first."""
        caption = table.find("caption")
        header = " ".join(" ".join(row) for row in rows[:3])
        previous = ""
        if self.builder.blocks and self.builder.blocks[-1].type in {
            "paragraph",
            "heading",
        }:
            previous = self.builder.blocks[-1].text[
                -300:
            ]  # the end: closest to the table
        return " ".join(
            [header, _text_of(caption) if isinstance(caption, Tag) else "", previous]
        )

    def walk(self, node: Tag, list_level: int = 0) -> None:
        buffer: list[str] = []

        def flush() -> None:
            text = _WS.sub(" ", "".join(buffer)).strip()
            buffer.clear()
            if text:
                self.builder.paragraph(text, index=self.next_index())

        for child in node.children:
            if isinstance(child, NavigableString):
                if isinstance(
                    child, (Comment, Doctype, CData, ProcessingInstruction, Declaration)
                ):
                    continue  # markup that is not text a reader sees
                buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            if _is_hidden(child):
                self.hidden[0] += 1
                continue
            name = child.name.lower()
            if name in _SKIP:
                continue
            if name == "br":
                buffer.append(" ")
            elif name in _HEADINGS:
                flush()
                self.builder.heading(
                    _text_of(child, self.hidden),
                    _HEADINGS[name],
                    index=self.next_index(),
                )
            elif name == "table":
                flush()
                self.table(child)
            elif name in {"ul", "ol"}:
                flush()
                self.list_block(child, list_level)
            elif name == "pre":
                flush()
                self.builder.paragraph(child.get_text(), index=self.next_index())
            elif name in _BLOCKS:
                flush()
                if self.has_block_children(child):
                    self.walk(child, list_level)
                else:
                    text = _text_of(child, self.hidden)
                    if text:
                        level = heading_level(child, text)
                        if level is None:
                            self.builder.paragraph(text, index=self.next_index())
                        else:
                            self.builder.heading(text, level, index=self.next_index())
            else:
                # inline element (span, a, b ...) or an unknown tag: its text joins the paragraph,
                # unless it wraps blocks (inline XBRL tags often do), in which case walk into it
                if self.has_block_children(child):
                    flush()
                    self.walk(child, list_level)
                else:
                    buffer.append(child.get_text())
        flush()

    @staticmethod
    def has_block_children(tag: Tag) -> bool:
        return any(
            isinstance(c, Tag)
            and (c.name.lower() in _BLOCKS or _Walker.has_block_children(c))
            for c in tag.children
        )

    def list_block(self, tag: Tag, level: int) -> None:
        for item in tag.find_all("li", recursive=False):
            if _is_hidden(item):
                self.hidden[0] += 1
                continue
            own: list[str] = []
            for c in item.children:
                if isinstance(c, NavigableString):
                    if not isinstance(
                        c, (Comment, Doctype, CData, ProcessingInstruction, Declaration)
                    ):
                        own.append(str(c))
                elif isinstance(c, Tag) and c.name not in {"ul", "ol"}:
                    if _is_hidden(c):
                        self.hidden[0] += 1
                    else:
                        own.append(_text_of(c, self.hidden))
            self.builder.list_item(
                _WS.sub(" ", "".join(own)).strip(), level=level, index=self.next_index()
            )
            for nested in item.find_all(["ul", "ol"], recursive=False):
                self.list_block(nested, level + 1)

    def table(self, tag: Tag) -> None:
        rows, merged = table_rows(tag, self.hidden)
        rows = clean_filing_table(rows)
        self.builder.table(
            rows,
            has_merged_cells=merged,
            units_from=self.units_context(tag, rows),
            index=self.next_index(),
        )


def parse_html(path: Path, settings: Settings) -> ParsedDocument:
    text, flags = decode_text(path.read_bytes())
    try:
        soup = BeautifulSoup(text, "html.parser")
        body = soup.body or soup
        builder = BlockBuilder()
        walker = _Walker(builder)
        walker.walk(body)
    except RecursionError as exc:
        raise ParseError(
            "html_too_deep", "The HTML is nested too deeply to read."
        ) from exc

    if walker.hidden[0]:
        flags.append(f"hidden_content_removed:{walker.hidden[0]}")
    flags += builder.document_flags()
    if not builder.blocks:
        flags.append("no_content")
    chars = sum(len(b.text) for b in builder.blocks)
    return ParsedDocument(
        kind="html",
        parser="html/bs4",
        blocks=builder.blocks,
        pages=[
            PageReport(
                index=1,
                label="document",
                status="ok" if builder.blocks else "empty",
                chars=chars,
            )
        ],
        flags=flags,
    )
