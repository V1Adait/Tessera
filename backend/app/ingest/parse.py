"""The front door of parsing: pick the parser for a file's kind, and decide what the result means.

`parse_file` runs the right parser. `decide_status` turns the result into the document's status:

    parsed        read cleanly
    needs_review  read, but some pages or blocks are suspect (low confidence, totals that do not
                  add up, pages that could not be read or still need OCR)
    needs_ocr     a scan: nothing could be read without OCR

A document is never silently trusted: anything suspect is named in `status_detail`.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.app.config import Settings
from backend.app.ingest.models import Block, DocKind, DocStatus, ParsedDocument
from backend.app.ingest.parse_docx import parse_docx
from backend.app.ingest.parse_html import parse_html
from backend.app.ingest.parse_pdf import ScanParser, parse_pdf
from backend.app.ingest.parse_pptx import parse_pptx
from backend.app.ingest.parse_text import parse_csv, parse_md, parse_txt
from backend.app.ingest.parse_xlsx import parse_xlsx

Parser = Callable[[Path, Settings], ParsedDocument]

PARSERS: dict[DocKind, Parser] = {
    "docx": parse_docx,
    "xlsx": parse_xlsx,
    "pptx": parse_pptx,
    "html": parse_html,
    "csv": parse_csv,
    "txt": parse_txt,
    "md": parse_md,
}

LOW_CONFIDENCE = (
    0.7  # a block at or below this is suspect (ragged table, wrong totals, maths)
)
_MATH_FLAGS = ("math_notation", "math_region", "garbled_glyphs")


def _is_math_suspect(block: Block) -> bool:
    return any(f.startswith(_MATH_FLAGS) for f in block.flags)


def _is_suspect(block: Block) -> bool:
    return block.confidence <= LOW_CONFIDENCE or any(
        f.startswith("table_total_mismatch") for f in block.flags
    )


_PAGE_COUNT_KINDS = {"pdf", "pptx", "xlsx"}


def parse_file(
    path: Path, kind: DocKind, settings: Settings, scan_parser: ScanParser | None = None
) -> ParsedDocument:
    """Parse a stored file. Raises ParseError if the file cannot be read, and FileNotFoundError
    if it is not there (checked first: the parsers wrap every open error as 'corrupt')."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if kind == "pdf":
        return parse_pdf(path, settings, scan_parser)
    return PARSERS[kind](path, settings)


def _pages(numbers: list[int], limit: int = 10) -> str:
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown + (", ..." if len(numbers) > limit else "")


def decide_status(parsed: ParsedDocument) -> tuple[DocStatus, str | None]:
    """The document's status and, unless it is clean, a plain-language reason."""
    needs_ocr = [p.index for p in parsed.pages if p.status == "needs_ocr"]
    failed = [p.index for p in parsed.pages if p.status == "failed"]

    if not parsed.blocks:
        if needs_ocr and len(needs_ocr) == len(parsed.pages):
            return (
                "needs_ocr",
                f"scanned document: all {len(needs_ocr)} page(s) need OCR",
            )
        if failed:
            return (
                "needs_review",
                f"no content could be read; page(s) {_pages(failed)} failed",
            )
        return "needs_review", "no readable content was found"

    math = [b for b in parsed.blocks if _is_math_suspect(b)]
    suspect = [b for b in parsed.blocks if _is_suspect(b) and not _is_math_suspect(b)]
    problems: list[str] = []
    if needs_ocr:
        problems.append(f"page(s) {_pages(needs_ocr)} are scans that need OCR")
    if failed:
        problems.append(f"page(s) {_pages(failed)} could not be read")
    if math:
        problems.append(
            f"{len(math)} block(s) contain mathematical notation or garbled symbols that did not extract reliably"
        )
    if suspect:
        problems.append(
            f"{len(suspect)} block(s) are low confidence or have totals that do not add up"
        )
    if problems:
        return "needs_review", "; ".join(problems)
    return "parsed", None


def build_report(parsed: ParsedDocument) -> dict[str, Any]:
    """What gets stored beside the blocks: per page/sheet/slide status, flags, block counts."""
    suspect = sum(_is_suspect(b) and not _is_math_suspect(b) for b in parsed.blocks)
    math = sum(_is_math_suspect(b) for b in parsed.blocks)
    return {
        "pages": [p.model_dump() for p in parsed.pages],
        "flags": parsed.flags,
        "block_counts": dict(Counter(b.type for b in parsed.blocks)),
        "suspect_blocks": suspect,
        "math_blocks": math,
    }


def page_count(parsed: ParsedDocument) -> int | None:
    """PDF pages, slides or sheets. Word, HTML and text have no pages worth counting."""
    return len(parsed.pages) if parsed.kind in _PAGE_COUNT_KINDS else None
