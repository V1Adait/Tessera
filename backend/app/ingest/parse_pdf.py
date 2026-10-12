"""PDF parser, built on pdfplumber (MIT). Born-digital PDFs only need their text layer, so no OCR.

Per page it decides what the page is, then reads it:

  probe      real text? mostly one big image (a scan)?  -> scanned pages are flagged needs_ocr
  tables     ruled-line detection first, then text-alignment detection for borderless statements,
             with a filter that rejects the false tables the second method invents on plain text
  figures    raster images and vector charts: caption and any real labels are kept, the data
             points are not read (the block says so)
  text       the rest, in reading order (two-column pages are split at their gutter), grouped
             into headings, list items and paragraphs
  document   running headers and footers are dropped, tables that continue over a page break are
             stitched together, heading levels come from font sizes

Scanned pages are not read here. `scan_parser` is the hook where an OCR engine plugs in later.
"""

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pdfplumber

from backend.app.config import Settings
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import Block, PageReport, ParsedDocument, ParseError
from backend.app.ingest.tables import (
    detect_header_rows,
    is_data_cell,
    looks_like_year,
    pad_rows,
)
from backend.app.ingest.text import join_lines

logging.getLogger("pdfminer").setLevel(
    logging.ERROR
)  # pdfminer is noisy about odd files

_MIN_TEXT_CHARS = 40  # a page with less text than this and a big image is a scan
_BULLET = re.compile(r"^(?:[•◦▪▫‣⁃●○■□*]|[-–—])\s+|^\(?(?:\d{1,3}|[a-zA-Z])[.)]\s+")
_CAPTION = re.compile(
    r"^\s*(?:figure|fig\.?|chart|exhibit|graph|source)\b", re.IGNORECASE
)
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
_MAX_GRAPHIC_OBJECTS = (
    1200  # above this, vector-chart detection is skipped (clustering is quadratic)
)

BBox = tuple[float, float, float, float]  # x0, top, x1, bottom


class ScanParser(Protocol):
    """The hook for an OCR engine: read the given (1-based) pages of a scanned PDF into blocks."""

    def parse_pages(self, path: Path, pages: list[int]) -> list[Block]: ...


# --- what a page yields before it becomes blocks --------------------------------------------


@dataclass
class _Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool
    column: int = 0


@dataclass
class _Element:
    """One thing on a page, in reading order: a paragraph, heading, list item, table or figure."""

    kind: str  # heading | paragraph | list_item | table | figure
    page: int
    top: float
    bottom: float
    x0: float
    column: int = 0
    subtype: str = ""  # figures: 'raster' or 'vector'
    text: str = ""
    size: float = 0.0
    rows: list[list[str]] = field(default_factory=list)
    confidence: float = 1.0
    labels: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    page_height: float = 0.0
    bbox: BBox | None = None
    page_end: int | None = None
    merged: bool = False


@dataclass
class _Page:
    number: int
    height: float
    status: str = "ok"
    flags: list[str] = field(default_factory=list)
    lines: list[_Line] = field(default_factory=list)
    others: list[_Element] = field(default_factory=list)  # tables and figures
    chars: int = 0
    gutter: float | None = None


# --- geometry helpers -----------------------------------------------------------------------


def _center(obj: dict[str, Any]) -> tuple[float, float]:
    return (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2


def _inside(point: tuple[float, float], box: BBox) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a: BBox, b: BBox) -> float:
    """Share of the smaller box covered by the other."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller else 0.0


def _obj_box(obj: dict[str, Any]) -> BBox:
    return (obj["x0"], obj["top"], obj["x1"], obj["bottom"])


# --- lines ----------------------------------------------------------------------------------


def _read_lines(page: Any) -> list[_Line]:
    lines: list[_Line] = []
    for raw in page.extract_text_lines(layout=False, strip=True, return_chars=True):
        text = str(raw["text"]).strip()
        if not text:
            continue
        chars = raw["chars"]
        sizes = [c["size"] for c in chars if c.get("size")]
        bold = sum(
            bool(_BOLD_FONT.search(str(c.get("fontname", "")))) for c in chars
        ) >= 0.6 * max(len(chars), 1)
        lines.append(
            _Line(
                text=text,
                x0=raw["x0"],
                x1=raw["x1"],
                top=raw["top"],
                bottom=raw["bottom"],
                size=round(statistics.median(sizes), 1) if sizes else 0.0,
                bold=bold,
            )
        )
    return lines


def _find_gutter(page: Any) -> float | None:
    """x position of a blank vertical strip separating two columns of text, or None.

    A real gutter has words on both sides all the way down and NO word crossing it, so a page
    with a full-width title is (correctly) not split: its text order is flagged instead."""
    words = page.extract_words()
    if len(words) < 80:
        return None
    width = float(page.width)
    best: tuple[float, float] | None = None  # (gap width, centre)
    x = width * 0.3
    while x < width * 0.7:
        if not any(w["x0"] <= x <= w["x1"] for w in words):
            start = x
            while x < width * 0.7 and not any(w["x0"] <= x <= w["x1"] for w in words):
                x += 1.0
            if x - start >= 8.0 and (best is None or x - start > best[0]):
                best = (x - start, (start + x) / 2)
        x += 1.0
    if best is None:
        return None
    centre = best[1]
    left = sum(w["x1"] < centre for w in words)
    right = sum(w["x0"] > centre for w in words)
    return centre if min(left, right) >= 0.25 * len(words) else None


# --- tables ---------------------------------------------------------------------------------


def _is_number_like(cell: str) -> bool:
    return is_data_cell(cell) or looks_like_year(cell)


def _accept_ruled(rows: list[list[str]]) -> bool:
    """A table found by its ruling lines: strong evidence, so only reject degenerate ones."""
    cells = [c for row in rows for c in row if c is not None]
    if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2 or not cells:
        return False
    return sum(bool(c.strip()) for c in cells) / len(cells) >= 0.25


def _accept_aligned(rows: list[list[str]]) -> bool:
    """A table found only by text alignment: pdfplumber also 'finds' these in ordinary
    paragraphs, so demand figures: mostly rows with several cells and plenty of numbers."""
    filled = [[c.strip() for c in row if c and c.strip()] for row in rows]
    non_empty_rows = [r for r in filled if r]
    if len(non_empty_rows) < 3 or max((len(row) for row in rows), default=0) > 30:
        return False
    multi = sum(len(r) >= 2 for r in non_empty_rows) / len(non_empty_rows)
    cells = [c for r in non_empty_rows for c in r]
    numeric = sum(_is_number_like(c) for c in cells) / len(cells)
    return multi >= 0.6 and numeric >= 0.4


def _extract(table: Any) -> list[list[str]]:
    try:
        return [[(c or "") for c in row] for row in table.extract()]
    except Exception:  # noqa: BLE001 - one bad table must not lose the page
        return []


def _detect_tables(page: Any, number: int) -> list[_Element]:
    height = float(page.height)
    found: list[_Element] = []
    for table in page.find_tables():
        rows = _extract(table)
        if _accept_ruled(rows):
            found.append(_table_element(number, table.bbox, rows, 0.95, height))
    aligned_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    for table in page.find_tables(aligned_settings):
        box: BBox = table.bbox
        if any(_overlap(box, f.bbox) > 0.3 for f in found if f.bbox):
            continue  # already found by its ruling lines
        rows = _extract(table)
        if _accept_aligned(rows):
            found.append(_table_element(number, box, rows, 0.85, height))
    return found


def _table_element(
    number: int, box: BBox, rows: list[list[str]], confidence: float, height: float
) -> _Element:
    return _Element(
        kind="table",
        page=number,
        top=box[1],
        bottom=box[3],
        x0=box[0],
        rows=rows,
        confidence=confidence,
        page_height=height,
        bbox=box,
    )


# --- figures --------------------------------------------------------------------------------


def _union_boxes(boxes: list[BBox], margin: float = 8.0) -> list[BBox]:
    """Merge boxes that touch or overlap (after growing each by `margin`) into clusters."""
    clusters = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[list[float]] = []
        for box in clusters:
            for other in merged:
                if (
                    box[0] - margin <= other[2]
                    and other[0] - margin <= box[2]
                    and box[1] - margin <= other[3]
                    and other[1] - margin <= box[3]
                ):
                    other[0], other[1] = min(other[0], box[0]), min(other[1], box[1])
                    other[2], other[3] = max(other[2], box[2]), max(other[3], box[3])
                    changed = True
                    break
            else:
                merged.append(box)
        clusters = merged
    return [(c[0], c[1], c[2], c[3]) for c in clusters]


def _detect_figures(page: Any, number: int, table_boxes: list[BBox]) -> list[_Element]:
    """Raster images and vector charts found on a page (captions and labels are filled in later)."""
    page_box: BBox = (0.0, 0.0, float(page.width), float(page.height))
    page_area = _area(page_box)
    height = float(page.height)
    figures: list[_Element] = []
    boxes: list[tuple[BBox, str]] = []

    for image in page.images:
        box = _obj_box(image)
        ratio = _area(box) / page_area if page_area else 0.0
        if 0.03 <= ratio < 0.6 and not any(_overlap(box, t) > 0.5 for t in table_boxes):
            boxes.append((box, "raster"))

    graphics = [*page.lines, *page.curves, *page.rects]
    if 10 <= len(graphics) <= _MAX_GRAPHIC_OBJECTS:
        usable = []
        for obj in graphics:
            box = _obj_box(obj)
            if _area(box) > 0.8 * page_area or any(
                _inside(_center(obj), t) for t in table_boxes
            ):
                continue
            usable.append(box)
        for cluster in _union_boxes(usable):
            members = sum(_inside(_center(o), cluster) for o in graphics)
            width, tall = cluster[2] - cluster[0], cluster[3] - cluster[1]
            if (
                members >= 10
                and _area(cluster) >= 0.04 * page_area
                and min(width, tall) >= 60
            ):
                boxes.append((cluster, "vector"))

    for box, kind in boxes:
        figures.append(
            _Element(
                kind="figure",
                page=number,
                top=box[1],
                bottom=box[3],
                x0=box[0],
                page_height=height,
                bbox=box,
                flags=["chart_data_not_extracted"]
                if kind == "vector"
                else ["image_content_not_read"],
                labels=[],
                subtype=kind,
            )
        )
    return figures


# --- one page -------------------------------------------------------------------------------


def _read_page(page: Any, number: int) -> _Page:
    result = _Page(number=number, height=float(page.height))
    text = page.extract_text() or ""
    result.chars = len(text.strip())
    page_area = float(page.width) * float(page.height)
    cover = (
        max((_area(_obj_box(i)) / page_area for i in page.images), default=0.0)
        if page_area
        else 0.0
    )

    if result.chars < _MIN_TEXT_CHARS and cover > 0.5:
        result.status = "needs_ocr"
        return result
    if (
        result.chars == 0
        and not page.images
        and not (page.lines or page.curves or page.rects)
    ):
        result.status = "empty"
        return result

    tables = _detect_tables(page, number)
    table_boxes: list[BBox] = [t.bbox for t in tables if t.bbox]
    figures = _detect_figures(page, number, table_boxes)
    regions = table_boxes + [f.bbox for f in figures if f.bbox]

    def keep(obj: dict[str, Any]) -> bool:
        return obj.get("object_type") != "char" or not any(
            _inside(_center(obj), r) for r in regions
        )

    body = page.filter(keep) if regions else page
    result.gutter = _find_gutter(body)
    if result.gutter is not None:
        result.flags.append("two_column_layout")
        left = body.crop((0, 0, result.gutter, float(page.height)), strict=False)
        right = body.crop(
            (result.gutter, 0, float(page.width), float(page.height)), strict=False
        )
        for column, part in enumerate((left, right)):
            for line in _read_lines(part):
                line.column = column
                result.lines.append(line)
    else:
        result.lines = _read_lines(body)

    # figures: read their real text (labels) and find their captions among the page's lines
    for figure in figures:
        assert figure.bbox is not None
        if figure.subtype == "vector":
            crop = page.crop(figure.bbox, strict=False)
            figure.labels = [ln.text for ln in _read_lines(crop)]
        caption = _find_caption(result.lines, figure.bbox)
        if caption is not None:
            result.lines.remove(caption)
            figure.text = caption.text
        figure.column = 0 if result.gutter is None or figure.x0 < result.gutter else 1

    unlabelled_images = [
        f for f in figures if "image_content_not_read" in f.flags and not f.text
    ]
    if unlabelled_images:
        result.flags.append(f"images_not_read:{len(unlabelled_images)}")
    result.others = [t for t in tables if t.rows] + [
        f for f in figures if f.text or f.labels
    ]
    if result.gutter is not None:
        for other in result.others:
            other.column = 0 if other.x0 < result.gutter else 1
    return result


def _find_caption(lines: list[_Line], box: BBox) -> _Line | None:
    for line in lines:
        near = (0 <= line.top - box[3] <= 40) or (0 <= box[1] - line.bottom <= 40)
        if near and _CAPTION.match(line.text) and box[0] - 40 <= line.x0 <= box[2]:
            return line
    return None


# --- document level: running headers and footers ----------------------------------------------


_MARGIN = 0.07  # running headers and footers sit in the outer 7% of the page; body text does not
_MAX_RUNNING_CHARS = 100


def _in_margin(line: _Line, height: float) -> bool:
    return line.top < _MARGIN * height or line.bottom > (1 - _MARGIN) * height


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _drop_running_lines(pages: list[_Page]) -> int:
    """Remove lines that repeat in the top or bottom margin of most pages (running headers,
    footers, page numbers). Returns how many lines were removed."""
    text_pages = [p for p in pages if p.lines]
    if len(text_pages) < 3:
        return 0
    seen: Counter[str] = Counter()
    for page in text_pages:
        keys = {
            _signature(line.text)
            for line in page.lines
            if _in_margin(line, page.height) and len(line.text) <= _MAX_RUNNING_CHARS
        }
        seen.update(keys)
    repeated = {
        key
        for key, count in seen.items()
        if count >= max(3, 0.5 * len(text_pages)) and key
    }
    removed = 0
    for page in text_pages:
        keep = []
        for line in page.lines:
            in_margin = (
                line.top < 0.12 * page.height or line.bottom > 0.88 * page.height
            )
            if in_margin and _signature(line.text) in repeated:
                removed += 1
            else:
                keep.append(line)
        page.lines = keep
    return removed


# --- text into paragraphs --------------------------------------------------------------------


def _body_size(pages: list[_Page]) -> float:
    weight: Counter[float] = Counter()
    for page in pages[:40]:
        for line in page.lines:
            weight[line.size] += len(line.text)
    return weight.most_common(1)[0][0] if weight else 0.0


def _is_heading(line: _Line, body: float) -> bool:
    words = len(line.text.split())
    if line.size >= body * 1.15 and len(line.text) <= 200:
        return True
    return (
        line.bold
        and words <= 15
        and len(line.text) <= 120
        and not line.text.rstrip().endswith((".", ",", ";"))
    )


def _paragraphs(page: _Page, body: float) -> list[_Element]:
    elements: list[_Element] = []
    current: list[_Line] = []
    kind = "paragraph"

    def flush() -> None:
        nonlocal current, kind
        if current:
            text = join_lines([line.text for line in current])
            first = current[0]
            elements.append(
                _Element(
                    kind=kind,
                    page=page.number,
                    top=first.top,
                    bottom=current[-1].bottom,
                    x0=first.x0,
                    column=first.column,
                    text=text,
                    size=max(line.size for line in current),
                    page_height=page.height,
                )
            )
        current = []
        kind = "paragraph"

    ordered = sorted(page.lines, key=lambda ln: (ln.column, ln.top, ln.x0))
    for line in ordered:
        previous = current[-1] if current else None
        if previous is not None:
            gap = line.top - previous.bottom
            line_height = max(previous.bottom - previous.top, 1.0)
            heading_line = _is_heading(line, body)
            if kind == "heading":
                # a heading that wraps onto a second line stays one heading
                new_block = not (
                    heading_line
                    and line.column == previous.column
                    and abs(line.size - previous.size) <= 0.5
                    and 0 <= gap <= 0.6 * line_height
                )
            else:
                new_block = (
                    line.column != previous.column
                    or gap > 0.9 * line_height
                    or gap < -line_height  # went back up: a different region
                    or heading_line
                    or bool(_BULLET.match(line.text))
                    or abs(line.size - previous.size) > 1.0
                )
            if new_block:
                flush()
        if not current:
            if _is_heading(line, body):
                kind = "heading"
            elif _BULLET.match(line.text):
                kind = "list_item"
        current.append(line)
    flush()
    return elements


def _heading_levels(elements: list[_Element], body: float) -> dict[float, int]:
    """Rank heading font sizes: the biggest is level 1. Headings as large as body text (only
    bold) rank after all the larger ones."""
    sizes = sorted(
        {e.size for e in elements if e.kind == "heading" and e.size > body * 1.02},
        reverse=True,
    )
    levels = {size: rank for rank, size in enumerate(sizes, 1)}
    levels[0.0] = min(len(sizes) + 1, 6)  # bold-only headings
    return levels


# --- stitching tables across pages ------------------------------------------------------------


def _stitch(elements: list[_Element]) -> list[_Element]:
    """Merge a table that ends near the bottom of a page with one that starts near the top of the
    next (same number of columns), dropping a repeated header row."""
    out: list[_Element] = []
    for element in elements:
        previous = out[-1] if out else None
        if (
            previous is not None
            and previous.kind == element.kind == "table"
            and element.page == (previous.page_end or previous.page) + 1
            and previous.bottom >= 0.7 * previous.page_height
            and element.top <= 0.3 * element.page_height
        ):
            a_rows, _ = pad_rows(previous.rows)
            b_rows, _ = pad_rows(element.rows)
            if a_rows and b_rows and len(a_rows[0]) == len(b_rows[0]):
                header = max(detect_header_rows(a_rows), 0)
                repeated = header and [c.strip() for c in b_rows[0]] == [
                    c.strip() for c in a_rows[0]
                ]
                if repeated:
                    b_rows = b_rows[header:]
                elif detect_header_rows(b_rows) > 0 and header == 0:
                    out.append(
                        element
                    )  # a different table that happens to start the next page
                    continue
                previous.rows = a_rows + b_rows
                previous.page_end = element.page
                previous.confidence = min(previous.confidence, element.confidence)
                previous.flags.append(
                    f"stitched_across_pages:{previous.page}-{element.page}"
                )
                continue
        out.append(element)
    return out


# --- the parser -----------------------------------------------------------------------------


def parse_pdf(
    path: Path, settings: Settings, scan_parser: ScanParser | None = None
) -> ParsedDocument:
    try:
        pdf = pdfplumber.open(path)
    except Exception as exc:
        if "password" in type(exc).__name__.lower() or "encrypt" in str(exc).lower():
            raise ParseError(
                "encrypted_pdf", "Password-protected PDFs are not supported."
            ) from exc
        raise ParseError("corrupt_pdf", "The PDF could not be opened.") from exc

    with pdf:
        count = len(pdf.pages)
        if count > settings.max_pdf_pages:
            raise ParseError(
                "too_many_pages",
                f"The PDF has {count} pages; the limit is {settings.max_pdf_pages}.",
            )
        pages: list[_Page] = []
        for number, page in enumerate(pdf.pages, 1):
            try:
                pages.append(_read_page(page, number))
            except Exception:
                logging.getLogger(__name__).exception(
                    "could not read PDF page %s", number
                )
                bad = _Page(number=number, height=float(page.height), status="failed")
                pages.append(bad)

    removed = _drop_running_lines(pages)
    body = _body_size(pages)
    elements: list[_Element] = []
    for record in pages:
        page_elements = _paragraphs(record, body) + record.others
        elements += sorted(page_elements, key=lambda e: (e.column, e.top, e.x0))
    levels = _heading_levels(elements, body)
    elements = _stitch(elements)

    builder = BlockBuilder()
    for element in elements:
        where: dict[str, Any] = {"page": element.page, "bbox": element.bbox}
        if element.page_end:
            where["page_end"] = element.page_end
        if element.kind == "heading":
            builder.heading(
                element.text, levels.get(element.size, levels[0.0]), page=element.page
            )
        elif element.kind == "list_item":
            builder.list_item(element.text, page=element.page)
        elif element.kind == "paragraph":
            builder.paragraph(element.text, page=element.page)
        elif element.kind == "table":
            previous = builder.blocks[-1].text[:300] if builder.blocks else ""
            builder.table(
                element.rows,
                confidence=element.confidence,
                units_from=previous,
                flags=element.flags,
                **where,
            )
        elif element.kind == "figure":
            builder.figure(element.text, element.labels, flags=element.flags, **where)

    flags: list[str] = []
    ocr_pages = [p.number for p in pages if p.status == "needs_ocr"]
    ocr_blocks: list[Block] = []
    if ocr_pages and scan_parser is not None:
        ocr_blocks = scan_parser.parse_pages(path, ocr_pages)
        for block in ocr_blocks:
            block.confidence = min(block.confidence, 0.8)
            block.flags.append("ocr")
        for record in pages:
            if record.number in ocr_pages:
                record.status, record.flags = "ok", [*record.flags, "ocr"]
    blocks = _merge_by_page(builder.blocks, ocr_blocks)

    if removed:
        flags.append(f"running_headers_removed:{removed}")
    flags += builder.document_flags()
    if not blocks:
        flags.append("no_content")
    reports = [
        PageReport(index=p.number, status=p.status, chars=p.chars, flags=p.flags)  # type: ignore[arg-type]
        for p in pages
    ]
    return ParsedDocument(
        kind="pdf", parser="pdf/pdfplumber", blocks=blocks, pages=reports, flags=flags
    )


def _merge_by_page(blocks: list[Block], extra: list[Block]) -> list[Block]:
    """Insert OCR blocks among the text blocks in page order (stable within a page)."""
    if not extra:
        return blocks
    return sorted([*blocks, *extra], key=lambda b: b.location.page or 0)
