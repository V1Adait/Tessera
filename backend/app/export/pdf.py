"""The matrix as a PDF: a summary table, then the sources behind every figure.

    page 1..   SUMMARY    companies down the side, questions across (split into blocks so the text
                          stays readable). Each cell shows its value and period, and says plainly when
                          it was computed, when it does not tie out, or when it was not found
    after      SOURCES    per company: every answer with the quote it came from, the file and page; how
                          computed figures were computed; the tie-out results; what was not answered

It reports what the matrix holds and invents nothing: an answer that was never verified is shown as
unverified, a missing one as missing. Pure function from data to bytes, no database, so it is tested
without one. PDF standard fonts cannot draw every character, so text is made safe first (see `_safe`).
"""

from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from typing import Any
from uuid import UUID
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.app.db.records import EntityRecord
from backend.app.matrix.models import CellRecord, ColumnRecord

MAX_ROWS = 200
MAX_COLUMNS = 60
COLUMNS_PER_BLOCK = 6
QUOTE_CHARS = 300
MAX_QUOTES = 3

INK = colors.HexColor("#37352f")
MUTED = colors.HexColor("#787774")
RULE = colors.HexColor("#e9e9e7")
AMBER = "#cb912f"
GREEN = "#1c3829"


class ExportTooLargeError(ValueError):
    """The matrix is too big to put in one readable PDF. The message says what to do."""


# --- text -------------------------------------------------------------------------------------------------------

_REPLACE = {
    "−": "-",  # minus sign
    "→": "->",
    "≥": ">=",
    "≤": "<=",
    " ": " ",
    " ": " ",
    " ": " ",
    "\u200b": "",
    "ƒ": "f",
}


def _safe(text: str) -> str:
    """Text the PDF's built-in font can draw. A character it has no glyph for becomes '?', so a
    document's odd symbol can never crash the export or print as a black box."""
    for old, new in _REPLACE.items():
        text = text.replace(old, new)
    return text.encode("cp1252", "replace").decode("cp1252")


def _p(text: str) -> str:
    """Text for a Paragraph: safe, and with <, > and & escaped so it is never read as markup."""
    return escape(_safe(text))


# --- styles -----------------------------------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "base", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK
    )
    return {
        "base": base,
        "small": ParagraphStyle(
            "small", parent=base, fontSize=7, leading=9, textColor=MUTED
        ),
        "title": ParagraphStyle(
            "title", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=3,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            spaceBefore=8,
            spaceAfter=2,
        ),
        "head": ParagraphStyle(
            "head", parent=base, fontName="Helvetica-Bold", textColor=MUTED
        ),
    }


# --- one cell -----------------------------------------------------------------------------------------------------

_STATUS_WORDS = {
    "not_found": "Not found",
    "not_applicable": "Does not apply",
    "conflicting": "Conflicting",
    "pending_or_error": "Could not be read",
}


def _worst_check(cell: CellRecord) -> str | None:
    statuses = {c.get("status") for c in cell.checks}
    for status in ("does_not_tie", "cannot_check", "ties"):
        if status in statuses:
            return status
    return None


def _cell_markup(cell: CellRecord | None) -> str:
    """What a summary cell says: the value, its period, and anything a reader should be warned about."""
    if cell is None:
        return f'<font color="{MUTED.hexval().replace("0x", "#")}">Not filled</font>'
    muted = MUTED.hexval().replace("0x", "#")
    if cell.status == "answered" and cell.display_value:
        parts = [f"<b>{_p(cell.display_value)}</b>"]
        if cell.period:
            parts.append(f'<font color="{muted}" size="7">{_p(cell.period)}</font>')
        notes = []
        if cell.derivation is not None:
            notes.append("computed")
        worst = _worst_check(cell)
        if worst == "does_not_tie":
            notes.append(f'<font color="{AMBER}">does not tie out</font>')
        elif worst == "ties":
            notes.append(f'<font color="{GREEN}">ties out</font>')
        if cell.stale:
            notes.append(f'<font color="{AMBER}">out of date</font>')
        if notes:
            parts.append(f'<font color="{muted}" size="7">({", ".join(notes)})</font>')
        return " ".join(parts)
    if cell.status == "low_confidence":
        shown = f": {_p(cell.display_value)}" if cell.display_value else ""
        return f'<font color="{AMBER}">Unverified{shown}</font>'
    word = _STATUS_WORDS.get(cell.status, cell.status)
    return f'<font color="{muted}">{_p(word)}</font>'


# --- the summary ------------------------------------------------------------------------------------------------


def _summary(
    rows: Sequence[EntityRecord],
    columns: Sequence[ColumnRecord],
    by_key: dict[tuple[UUID, UUID], CellRecord],
    st: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    if not rows or not columns:
        return [
            Paragraph(
                "There is nothing to show yet: add at least one company and one question.",
                st["base"],
            )
        ]
    flow: list[Any] = []
    first_width = 95.0
    blocks = [
        columns[i : i + COLUMNS_PER_BLOCK]
        for i in range(0, len(columns), COLUMNS_PER_BLOCK)
    ]
    for number, block in enumerate(blocks, start=1):
        if len(blocks) > 1:
            lo = (number - 1) * COLUMNS_PER_BLOCK + 1
            flow.append(
                Paragraph(
                    f"Questions {lo} to {lo + len(block) - 1} of {len(columns)}",
                    st["h3"],
                )
            )
        header = [Paragraph("Company", st["head"])] + [
            Paragraph(_p(c.name), st["head"]) for c in block
        ]
        body = [
            [Paragraph(_company_markup(r), st["base"])]
            + [
                Paragraph(_cell_markup(by_key.get((r.id, c.id))), st["base"])
                for c in block
            ]
            for r in rows
        ]
        each = (width - first_width) / len(block)
        table = Table(
            [header, *body], colWidths=[first_width] + [each] * len(block), repeatRows=1
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        flow += [table, Spacer(1, 8)]
    return flow


def _company_markup(row: EntityRecord) -> str:
    ticker = (
        f' <font color="#787774" size="7">{_p(row.ticker)}</font>' if row.ticker else ""
    )
    return f"<b>{_p(row.name)}</b>{ticker}"


# --- the sources --------------------------------------------------------------------------------------------------


_OPS = {
    "ratio": lambda n: f"{n[0]} / {n[1]}",
    "difference": lambda n: f"{n[0]} - {n[1]}",
    "sum": lambda n: " + ".join(n),
    "growth": lambda n: f"{n[0]} against {n[1]}, as growth",
    "net_ratio": lambda n: f"({n[0]} - {n[1]}) / {n[2]}",
}


def _formula(derivation: dict[str, Any]) -> str:
    names = [str(i.get("column", "?")) for i in derivation.get("inputs", [])]
    make = _OPS.get(str(derivation.get("op")))
    try:
        return make(names) if make else ", ".join(names)
    except IndexError:
        return ", ".join(names)


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _evidence_markup(cell: CellRecord, filenames: dict[UUID, str]) -> str:
    lines: list[str] = []
    if cell.derivation is not None:
        lines.append(f"<b>Computed:</b> {_p(_formula(cell.derivation))}")
        for item in cell.derivation.get("inputs", []):
            shown = item.get("display") or "-"
            period = f" ({item['period']})" if item.get("period") else ""
            lines.append(
                f"&nbsp;&nbsp;{_p(str(item.get('column', '?')))}: {_p(str(shown))}{_p(period)}"
            )
        for note in cell.derivation.get("notes", []):
            lines.append(f'<font color="{AMBER}">{_p(str(note))}</font>')
    for e in cell.evidence[:MAX_QUOTES]:
        where = filenames.get(e.document_id, "document")
        page = f", p. {e.page_start}" if e.page_start else ""
        lines.append(
            f'“{_p(_cut(e.quote, QUOTE_CHARS))}” <font color="#787774">{_p(where)}{_p(page)}</font>'
        )
    if len(cell.evidence) > MAX_QUOTES:
        lines.append(
            f'<font color="#787774">and {len(cell.evidence) - MAX_QUOTES} more quote(s)</font>'
        )
    for check in cell.checks:
        status = {
            "ties": "Ties out",
            "does_not_tie": "Does not tie out",
            "cannot_check": "Could not check",
        }.get(str(check.get("status")), str(check.get("status")))
        color = AMBER if check.get("status") == "does_not_tie" else "#787774"
        lines.append(
            f'<font color="{color}"><b>{_p(status)}:</b> {_p(str(check.get("detail", "")))}</font>'
        )
    if cell.status == "low_confidence" and cell.status_detail:
        lines.append(
            f'<font color="{AMBER}">Not verified: {_p(cell.status_detail)}</font>'
        )
    return "<br/>".join(lines) or "-"


def _sources(
    rows: Sequence[EntityRecord],
    columns: Sequence[ColumnRecord],
    by_key: dict[tuple[UUID, UUID], CellRecord],
    documents: dict[UUID, list[str]],
    filenames: dict[UUID, str],
    st: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    flow: list[Any] = [Paragraph("Sources", st["h2"])]
    flow.append(
        Paragraph(
            "For each company: every answer with the exact words it was taken from, the file and page. "
            "Computed figures show their formula and inputs. Anything that was not answered is listed last.",
            st["small"],
        )
    )
    first, second = 105.0, 130.0
    for row in rows:
        flow.append(Paragraph(_company_markup(row), st["h3"]))
        files = documents.get(row.id, [])
        flow.append(
            Paragraph(
                "Documents: "
                + (_p(", ".join(files)) if files else "none attached yet"),
                st["small"],
            )
        )
        table_rows: list[list[Any]] = []
        missing: list[str] = []
        for c in columns:
            cell = by_key.get((row.id, c.id))
            if cell is not None and cell.status in ("answered", "low_confidence"):
                table_rows.append(
                    [
                        Paragraph(f"<b>{_p(c.name)}</b>", st["base"]),
                        Paragraph(_cell_markup(cell), st["base"]),
                        Paragraph(_evidence_markup(cell, filenames), st["base"]),
                    ]
                )
            else:
                word = (
                    "not filled"
                    if cell is None
                    else _STATUS_WORDS.get(cell.status, cell.status).lower()
                )
                missing.append(f"{c.name} ({word})")
        if table_rows:
            table = Table(table_rows, colWidths=[first, second, width - first - second])
            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            flow.append(table)
        else:
            flow.append(Paragraph("No answers yet.", st["base"]))
        if missing:
            flow.append(Spacer(1, 3))
            flow.append(
                Paragraph("Not answered: " + _p("; ".join(missing)), st["small"])
            )
    return flow


# --- the document ------------------------------------------------------------------------------------------------


def build_pdf(
    workspace_name: str,
    rows: Sequence[EntityRecord],
    columns: Sequence[ColumnRecord],
    cells: Sequence[CellRecord],
    documents: dict[UUID, list[str]],
    filenames: dict[UUID, str],
    generated_at: datetime,
) -> bytes:
    """The workspace as PDF bytes. Raises ExportTooLargeError for a matrix too big to read on paper."""
    if len(rows) > MAX_ROWS or len(columns) > MAX_COLUMNS:
        raise ExportTooLargeError(
            f"This matrix has {len(rows)} companies and {len(columns)} questions; a PDF holds at most "
            f"{MAX_ROWS} and {MAX_COLUMNS}. Remove some, or export fewer."
        )
    st = _styles()
    page = landscape(A4)
    margin = 16 * mm
    width = page[0] - 2 * margin
    by_key = {(c.entity_id, c.column_id): c for c in cells}

    story: list[Any] = [
        Paragraph("TESSERA", st["small"]),
        Paragraph(_p(workspace_name), st["title"]),
        Paragraph(
            f"Generated {generated_at:%d %B %Y, %H:%M} UTC · {len(rows)} companies · "
            f"{len(columns)} questions",
            st["small"],
        ),
        Spacer(1, 4),
        Paragraph(
            "Every figure is quoted from the source documents and checked against them. Computed figures "
            "are calculated by code from other figures. A figure marked <i>Unverified</i> could not be "
            "backed by a quote and should not be relied on.",
            st["base"],
        ),
        Spacer(1, 6),
        *_summary(rows, columns, by_key, st, width),
    ]
    if rows and columns:
        story += [
            PageBreak(),
            *_sources(rows, columns, by_key, documents, filenames, st, width),
        ]

    name = _safe(workspace_name)

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, 9 * mm, f"Tessera - {name}")
        canvas.drawRightString(page[0] - margin, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = BytesIO()
    SimpleDocTemplate(
        out,
        pagesize=page,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=16 * mm,
        title=_safe(f"Tessera - {workspace_name}"),
        author="Tessera",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
