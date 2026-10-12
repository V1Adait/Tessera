"""Excel (.xlsx) parser.

Each sheet becomes a heading (the sheet name) followed by one table per block of data separated
by blank rows. Cell values are the values Excel last calculated; a formula that was never
calculated is reported, not guessed. Hidden sheets are still read but flagged.
"""

import datetime as dt
import zipfile
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

from backend.app.config import Settings
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import PageReport, ParsedDocument, ParseError

_MAX_ROWS = 100_000
_MAX_COLS = 200
_MAX_CELLS = 2_000_000
_GAP_ROWS = 2  # this many empty rows in a row end a table
_FORMULA_CHECK_LIMIT_BYTES = 20 * 1024 * 1024


def _decimals(number_format: str) -> int:
    """Digits after the decimal point that a number format shows: '#,##0.00' -> 2, '0.0%' -> 1."""
    section = number_format.split(";")[0]
    if "." not in section:
        return 0
    count = 0
    for ch in section.split(".", 1)[1]:
        if ch not in "0#":
            break
        count += 1
    return count


def format_cell(cell: object) -> str:
    """A cell as a reader sees it: percentages, thousands separators and dates are formatted
    the way the cell's number format says, in the common cases."""
    value = getattr(cell, "value", None)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, dt.datetime):
        return (
            value.date().isoformat() if value.time() == dt.time() else value.isoformat()
        )
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        fmt = str(getattr(cell, "number_format", "General") or "General")
        places = _decimals(fmt)
        if "%" in fmt:
            return f"{value * 100:.{places}f}%"
        if "#,##0" in fmt or "#,###" in fmt:
            return f"{value:,.{places}f}"
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value)
    return str(value)


def _regions(grid: list[list[str]]) -> list[tuple[int, int]]:
    """(first_row, last_row) of each block of data, split where there are blank rows."""
    regions: list[tuple[int, int]] = []
    start: int | None = None
    blanks = 0
    for r, row in enumerate(grid):
        if any(row):
            if start is None:
                start = r
            blanks = 0
        elif start is not None:
            blanks += 1
            if blanks >= _GAP_ROWS:
                regions.append((start, r - blanks))
                start = None
                blanks = 0
    if start is not None:
        last = len(grid) - 1
        while last >= start and not any(grid[last]):
            last -= 1
        regions.append((start, last))
    return regions


def _formula_gaps(path: Path, limit: int = 1000) -> dict[str, int]:
    """Per sheet: formula cells that have no stored result (the file was never calculated by Excel).
    Reads the workbook twice (formulas, then values), once per file."""
    if path.stat().st_size > _FORMULA_CHECK_LIMIT_BYTES:
        return {}
    with path.open("rb") as handle:
        formulas = openpyxl.load_workbook(handle, data_only=False, read_only=False)
    with path.open("rb") as handle:
        values = openpyxl.load_workbook(handle, data_only=True, read_only=False)
    gaps: dict[str, int] = {}
    for sheet in formulas.worksheets:
        count = 0
        for row_f, row_v in zip(
            sheet.iter_rows(), values[sheet.title].iter_rows(), strict=False
        ):
            for cf, cv in zip(row_f, row_v, strict=False):
                if (
                    isinstance(cf.value, str)
                    and cf.value.startswith("=")
                    and cv.value is None
                ):
                    count += 1
        if count:
            gaps[sheet.title] = min(count, limit)
    return gaps


def parse_xlsx(path: Path, settings: Settings) -> ParsedDocument:
    try:
        # a file OBJECT, not a path: openpyxl rejects paths that do not end in .xlsx, and our
        # uploads are stored under their hash with no extension
        with path.open("rb") as handle:
            workbook = openpyxl.load_workbook(handle, data_only=True, read_only=False)
    except Exception as exc:
        raise ParseError("corrupt_xlsx", "The Excel file could not be opened.") from exc

    builder = BlockBuilder()
    pages: list[PageReport] = []
    flags: list[str] = []
    total_cells = 0
    formula_gaps = _formula_gaps(path)

    for position, sheet in enumerate(workbook.worksheets, 1):
        sheet_flags: list[str] = []
        if sheet.sheet_state != "visible":
            sheet_flags.append("hidden_sheet")
        max_row = min(sheet.max_row or 0, _MAX_ROWS)
        max_col = min(sheet.max_column or 0, _MAX_COLS)
        if (sheet.max_row or 0) > _MAX_ROWS or (sheet.max_column or 0) > _MAX_COLS:
            sheet_flags.append("truncated")
        total_cells += max_row * max_col
        if total_cells > _MAX_CELLS:
            raise ParseError(
                "workbook_too_large", "The workbook has too many cells to read."
            )

        grid = [
            [format_cell(cell) for cell in row]
            for row in sheet.iter_rows(
                min_row=1, max_row=max_row, min_col=1, max_col=max_col
            )
        ]
        hidden_rows = sum(1 for dim in sheet.row_dimensions.values() if dim.hidden)
        if hidden_rows:
            sheet_flags.append(f"hidden_rows:{hidden_rows}")
        merged_ranges = list(sheet.merged_cells.ranges)

        regions = _regions(grid)
        chars_before = sum(len(b.text) for b in builder.blocks)
        if regions:
            builder.heading(sheet.title, 1, sheet=sheet.title, index=position)
        for first, last in regions:
            block_rows = grid[first : last + 1]
            columns = [c for c in range(max_col) if any(row[c] for row in block_rows)]
            if not columns:
                continue
            low, high = columns[0], columns[-1]
            builder.table(
                [row[low : high + 1] for row in block_rows],
                has_merged_cells=any(
                    rng.min_row <= last + 1 and rng.max_row >= first + 1
                    for rng in merged_ranges
                ),
                sheet=sheet.title,
                cell_range=f"{get_column_letter(low + 1)}{first + 1}:{get_column_letter(high + 1)}{last + 1}",
                flags=list(sheet_flags),
            )

        if sheet.title in formula_gaps:
            sheet_flags.append(f"formula_without_result:{formula_gaps[sheet.title]}")
        chars = sum(len(b.text) for b in builder.blocks) - chars_before
        pages.append(
            PageReport(
                index=position,
                label=sheet.title,
                status="ok" if regions else "empty",
                chars=chars,
                flags=sheet_flags,
            )
        )

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if any(n.startswith("xl/charts/") for n in names):
            flags.append("charts_not_read")
        if any(n.startswith("xl/media/") for n in names):
            flags.append("images_not_read")
    except zipfile.BadZipFile:
        pass
    flags += builder.document_flags()
    if not builder.blocks:
        flags.append("no_content")
    return ParsedDocument(
        kind="xlsx",
        parser="xlsx/openpyxl",
        blocks=builder.blocks,
        pages=pages,
        flags=flags,
    )
