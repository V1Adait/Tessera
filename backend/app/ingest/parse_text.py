"""Parsers for plain text, Markdown and CSV. No libraries: the formats are simple, and reading
them by hand keeps every rule visible."""

import csv
import io
import re
from pathlib import Path

from backend.app.config import Settings
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import PageReport, ParsedDocument, ParseError
from backend.app.ingest.text import join_lines

_MAX_CSV_ROWS = 200_000
_COLUMNAR = re.compile(r"\S {3,}\S")  # words separated by a wide gap: aligned columns


def decode_text(data: bytes) -> tuple[str, list[str]]:
    """Bytes to text: UTF-8 first, then Windows-1252 (common in old exports). Returns the flags
    to attach if the encoding had to be guessed."""
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), ["decoded_as_cp1252"]
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), ["decoding_errors"]


def _finish(
    kind: str, parser: str, builder: BlockBuilder, flags: list[str]
) -> ParsedDocument:
    chars = sum(len(block.text) for block in builder.blocks)
    return ParsedDocument(
        kind=kind,  # type: ignore[arg-type]
        parser=parser,
        blocks=builder.blocks,
        pages=[
            PageReport(
                index=1,
                label="document",
                status="ok" if builder.blocks else "empty",
                chars=chars,
            )
        ],
        flags=[
            *flags,
            *builder.document_flags(),
            *([] if builder.blocks else ["no_content"]),
        ],
    )


# --- plain text -------------------------------------------------------------------------


def parse_txt(path: Path, settings: Settings) -> ParsedDocument:
    text, flags = decode_text(path.read_bytes())
    builder = BlockBuilder()
    for index, chunk in enumerate(re.split(r"\n\s*\n", text.replace("\r\n", "\n"))):
        lines = chunk.split("\n")
        columnar = (
            len(lines) >= 3
            and sum(bool(_COLUMNAR.search(line)) for line in lines) >= len(lines) / 2
        )
        if columnar:
            # aligned columns (old plain-text filings): reflowing would destroy them, so keep the lines
            block = builder.paragraph("\n".join(lines), index=index)
            if block is not None:
                block.flags.append("columnar_text_kept_as_is")
        else:
            builder.paragraph(join_lines(lines), index=index)
    return _finish("txt", "text", builder, flags)


# --- Markdown ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_md(path: Path, settings: Settings) -> ParsedDocument:
    text, flags = decode_text(path.read_bytes())
    lines = text.replace("\r\n", "\n").split("\n")
    builder = BlockBuilder()
    paragraph: list[str] = []
    paragraph_start = 0

    def flush() -> None:
        if paragraph:
            builder.paragraph(join_lines(paragraph), index=paragraph_start)
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            flush()
            fence = stripped[:3]
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            block = builder.paragraph("\n".join(code), index=i)
            if block is not None:
                block.flags.append("code_block")
        elif not stripped:
            flush()
        elif (match := _HEADING.match(line)) is not None:
            flush()
            builder.heading(match.group(2), len(match.group(1)), index=i)
        elif (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and _TABLE_SEPARATOR.match(lines[i + 1])
        ):
            flush()
            rows = [_split_pipe_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_pipe_row(lines[i]))
                i += 1
            builder.table(rows, index=i)
            continue
        elif (match := _LIST.match(line)) is not None:
            flush()
            builder.list_item(match.group(2), level=len(match.group(1)) // 2, index=i)
        else:
            if not paragraph:
                paragraph_start = i
            paragraph.append(line)
        i += 1
    flush()
    return _finish("md", "markdown", builder, flags)


# --- CSV --------------------------------------------------------------------------------


def column_letters(number: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA (spreadsheet column names)."""
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def parse_csv(path: Path, settings: Settings) -> ParsedDocument:
    text, flags = decode_text(path.read_bytes())
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
            text[:8192], delimiters=",;\t|"
        )
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows: list[list[str]] = []
    truncated = False
    try:
        for row in reader:
            if len(rows) >= _MAX_CSV_ROWS:
                truncated = True
                break
            rows.append(row)
    except csv.Error as exc:
        raise ParseError("bad_csv", "The CSV file could not be read.") from exc

    builder = BlockBuilder()
    width = max((len(r) for r in rows), default=0)
    extra = ["truncated_rows"] if truncated else []
    builder.table(
        rows, flags=extra, cell_range=f"A1:{column_letters(max(width, 1))}{len(rows)}"
    )
    return _finish("csv", "csv", builder, flags)
