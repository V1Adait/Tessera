"""Word (.docx) parser.

python-docx's own `paragraph.text` silently skips text inside tracked insertions, so text is
read straight from the XML instead, where it is also possible to skip what a reader cannot see
(runs marked hidden). Text boxes, footnotes and comments are not read, and the document says so.
"""

import re
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table

from backend.app.config import Settings
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import PageReport, ParsedDocument, ParseError

_MAX_TABLE_CELLS = 200_000
_HEADING = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)
_REAL_FOOTNOTE = re.compile(r"<w:footnote\b(?![^>]*w:type=)[^>]*>")


def _in_text_box(element: object) -> bool:
    return any(a.tag == qn("w:txbxContent") for a in element.iterancestors())  # type: ignore[attr-defined]


def paragraph_text(paragraph: object) -> tuple[str, int, bool]:
    """(text, hidden runs skipped, whether a text box was skipped) for one <w:p> element."""
    parts: list[str] = []
    hidden = 0
    textbox = False
    for run in paragraph.iter(qn("w:r")):  # type: ignore[attr-defined]
        if _in_text_box(run):
            textbox = True
            continue
        properties = run.find(qn("w:rPr"))
        if properties is not None and properties.find(qn("w:vanish")) is not None:
            hidden += 1
            continue
        for child in run:
            if child.tag == qn("w:t"):
                parts.append(child.text or "")
            elif child.tag == qn("w:tab") or child.tag in {qn("w:br"), qn("w:cr")}:
                parts.append(" ")
            elif child.tag == qn("w:noBreakHyphen"):
                parts.append("-")
    return "".join(parts), hidden, textbox


class _Reader:
    def __init__(self, document: object) -> None:
        self.document = document
        self.builder = BlockBuilder()
        self.hidden_runs = 0
        self.textbox_seen = False
        self.counter = 0

    def next_index(self) -> int:
        self.counter += 1
        return self.counter

    def text(self, paragraph: object) -> str:
        text, hidden, textbox = paragraph_text(paragraph)
        self.hidden_runs += hidden
        self.textbox_seen = self.textbox_seen or textbox
        return text

    def style_name(self, paragraph: object) -> str:
        properties = paragraph.find(qn("w:pPr"))  # type: ignore[attr-defined]
        if properties is None:
            return ""
        style = properties.find(qn("w:pStyle"))
        if style is None:
            return ""
        style_id = style.get(qn("w:val")) or ""
        try:
            found = self.document.styles[style_id]  # type: ignore[attr-defined]
            return str(found.name or style_id)
        except KeyError:
            return style_id

    def list_level(self, paragraph: object) -> int | None:
        properties = paragraph.find(qn("w:pPr"))  # type: ignore[attr-defined]
        numbering = properties.find(qn("w:numPr")) if properties is not None else None
        if numbering is None:
            return None
        level = numbering.find(qn("w:ilvl"))
        try:
            return int(level.get(qn("w:val"))) if level is not None else 0
        except TypeError, ValueError:
            return 0

    def paragraph(self, element: object) -> None:
        text = self.text(element)
        name = self.style_name(element)
        index = self.next_index()
        if name.lower() == "title":
            self.builder.heading(text, 1, index=index)
        elif (match := _HEADING.match(name)) is not None:
            self.builder.heading(text, int(match.group(1)), index=index)
        elif (level := self.list_level(element)) is not None or name.lower().startswith(
            "list"
        ):
            self.builder.list_item(text, level=level or 0, index=index)
        else:
            self.builder.paragraph(text, index=index)

    def table(self, element: object) -> None:
        table = Table(element, self.document)  # type: ignore[arg-type]
        rows: list[list[str]] = []
        merged = False
        seen: dict[
            int, object
        ] = {}  # keeps the cell objects alive so id() cannot be reused
        cells_total = 0
        for row in table.rows:
            out: list[str] = []
            for cell in row.cells:
                cells_total += 1
                if cells_total > _MAX_TABLE_CELLS:
                    raise ParseError(
                        "table_too_large",
                        "A table in the document is too large to read.",
                    )
                tc = cell._tc
                if id(tc) in seen:
                    out.append("")  # the continuation of a merged cell
                    merged = True
                    continue
                seen[id(tc)] = tc
                lines = []
                for paragraph in tc.iter(qn("w:p")):
                    text, hidden, _ = paragraph_text(paragraph)
                    self.hidden_runs += hidden
                    lines.append(text)
                out.append(" ".join(line for line in lines if line.strip()))
            rows.append(out)
        previous = self.builder.blocks[-1].text[:300] if self.builder.blocks else ""
        first_row = " ".join(rows[0]) if rows else ""
        self.builder.table(
            rows,
            has_merged_cells=merged,
            units_from=f"{previous} {first_row}",
            index=self.next_index(),
        )

    def walk(self, container: object) -> None:
        for child in container.iterchildren():  # type: ignore[attr-defined]
            if child.tag == qn("w:p"):
                self.paragraph(child)
            elif child.tag == qn("w:tbl"):
                self.table(child)
            elif child.tag == qn(
                "w:sdt"
            ):  # a content control: its content is a normal body
                content = child.find(qn("w:sdtContent"))
                if content is not None:
                    self.walk(content)


def _extra_flags(path: Path) -> list[str]:
    """Things inside the file that this parser does not read, so the document can say so."""
    flags: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "word/footnotes.xml" in names:
                xml = archive.read("word/footnotes.xml")[:5_000_000].decode(
                    "utf-8", "replace"
                )
                if _REAL_FOOTNOTE.search(xml):
                    flags.append("footnotes_not_read")
            if "word/comments.xml" in names:
                flags.append("comments_not_read")
            body = archive.read("word/document.xml")[:50_000_000].decode(
                "utf-8", "replace"
            )
            if "<w:ins " in body or "<w:del " in body:
                flags.append("tracked_changes_present")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ParseError("corrupt_docx", "The Word file could not be opened.") from exc
    return flags


def parse_docx(path: Path, settings: Settings) -> ParsedDocument:
    try:
        document = Document(str(path))
    except Exception as exc:
        raise ParseError("corrupt_docx", "The Word file could not be opened.") from exc

    reader = _Reader(document)
    reader.walk(document.element.body)

    flags = _extra_flags(path)
    if reader.textbox_seen:
        flags.append("text_boxes_not_read")
    if reader.hidden_runs:
        flags.append(f"hidden_text_removed:{reader.hidden_runs}")
    flags += reader.builder.document_flags()
    blocks = reader.builder.blocks
    if not blocks:
        flags.append("no_content")
    return ParsedDocument(
        kind="docx",
        parser="docx/python-docx",
        blocks=blocks,
        pages=[
            PageReport(
                index=1,
                label="document",
                status="ok" if blocks else "empty",
                chars=sum(len(b.text) for b in blocks),
            )
        ],
        flags=flags,
    )
