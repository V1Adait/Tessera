"""BlockBuilder: how every parser turns what it reads into clean, located Blocks.

It owns the rules the parsers must not each reinvent: text is cleaned (and invisible characters
counted), empty blocks are dropped, headings form a section path, and tables are padded, trimmed
and checked.
"""

from backend.app.ingest.models import (
    Block,
    BlockType,
    Location,
    TableData,
    render_table,
)
from backend.app.ingest.tables import (
    detect_header_rows,
    drop_empty_edges,
    find_units,
    pad_rows,
    total_mismatches,
)
from backend.app.ingest.text import (
    clean_text,
    flag_for_removed,
    math_ratio,
    stray_glyphs,
)


class BlockBuilder:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._headings: list[tuple[int, str]] = []
        self.invisible_removed = 0
        self.garbled_blocks = 0  # blocks with stray glyphs (a font mapping accident)
        self.math_blocks = 0  # blocks that are mostly mathematical notation

    # --- helpers --------------------------------------------------------------------------

    @property
    def section_path(self) -> list[str]:
        return [title for _, title in self._headings]

    def _clean(self, text: str) -> tuple[str, list[str]]:
        cleaned, removed = clean_text(text)
        self.invisible_removed += removed
        return cleaned, flag_for_removed(removed)

    def _assess(self, text: str, *, check_math: bool = True) -> tuple[list[str], float]:
        """Flags and a confidence ceiling for text that did not extract reliably."""
        flags: list[str] = []
        ceiling = 1.0
        stray = stray_glyphs(text)
        if stray:
            flags.append(f"garbled_glyphs:{stray}")
            ceiling = 0.6
            self.garbled_blocks += 1
        if check_math and len(text) >= 4 and math_ratio(text) >= 0.2:
            flags.append("math_notation")
            ceiling = min(ceiling, 0.7)
            self.math_blocks += 1
        return flags, ceiling

    def _location(self, **where: object) -> Location:
        return Location(section_path=self.section_path, **where)  # type: ignore[arg-type]

    # --- text blocks ----------------------------------------------------------------------

    def heading(self, text: str, level: int, **where: object) -> Block | None:
        cleaned, flags = self._clean(text)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return None
        level = max(1, min(level, 6))
        while self._headings and self._headings[-1][0] >= level:
            self._headings.pop()
        self._headings.append((level, cleaned[:200]))
        extra, ceiling = self._assess(cleaned, check_math=False)
        block = Block(
            type="heading",
            text=cleaned,
            level=level,
            location=self._location(**where),
            flags=[*flags, *extra],
            confidence=ceiling,
        )
        self.blocks.append(block)
        return block

    def paragraph(
        self,
        text: str,
        *,
        confidence: float | None = None,
        flags: list[str] | None = None,
        **where: object,
    ) -> Block | None:
        return self._text_block("paragraph", text, where, confidence, flags)

    def list_item(
        self, text: str, level: int | None = None, **where: object
    ) -> Block | None:
        block = self._text_block("list_item", text, where, None, None)
        if block is not None:
            block.level = level
        return block

    def _text_block(
        self,
        kind: BlockType,
        text: str,
        where: dict[str, object],
        confidence: float | None,
        extra_flags: list[str] | None,
    ) -> Block | None:
        cleaned, flags = self._clean(text)
        if not cleaned:
            return None
        found, ceiling = self._assess(cleaned)
        block = Block(
            type=kind,
            text=cleaned,
            location=self._location(**where),
            flags=[*flags, *(extra_flags or []), *found],
            confidence=min(ceiling, confidence if confidence is not None else 1.0),
        )
        self.blocks.append(block)
        return block

    # --- tables and figures ---------------------------------------------------------------

    def table(
        self,
        rows: list[list[str]],
        *,
        has_merged_cells: bool = False,
        confidence: float = 1.0,
        units_from: str | None = None,
        flags: list[str] | None = None,
        **where: object,
    ) -> Block | None:
        """Add a table. `rows` are raw cell strings; they are cleaned, padded and checked here."""
        removed = 0
        cleaned_rows: list[list[str]] = []
        for row in rows:
            cleaned_row = []
            for cell in row:
                text, count = clean_text(cell or "")
                removed += count
                cleaned_row.append(" ".join(text.split()))
            cleaned_rows.append(cleaned_row)
        self.invisible_removed += removed

        cleaned_rows, ragged = pad_rows(cleaned_rows)
        cleaned_rows = drop_empty_edges(cleaned_rows)
        if not cleaned_rows:
            return None

        header_rows = detect_header_rows(cleaned_rows)
        all_flags = [*(flags or []), *flag_for_removed(removed)]
        if ragged:
            all_flags.append("ragged_rows")
            confidence = min(confidence, 0.7)
        stray = sum(stray_glyphs(cell) for row in cleaned_rows for cell in row)
        if stray:
            all_flags.append(f"garbled_glyphs:{stray}")
            confidence = min(confidence, 0.6)
            self.garbled_blocks += 1
        mismatches = total_mismatches(cleaned_rows, header_rows)
        if mismatches:
            all_flags += [f"table_total_mismatch:r{r}c{c}" for r, c in mismatches[:3]]
            confidence = min(confidence, 0.5)

        data = TableData(
            rows=cleaned_rows,
            header_rows=header_rows,
            units=find_units(units_from) if units_from else None,
            has_merged_cells=has_merged_cells,
        )
        block = Block(
            type="table",
            text=render_table(cleaned_rows),
            location=self._location(**where),
            table=data,
            confidence=confidence,
            flags=all_flags,
        )
        self.blocks.append(block)
        return block

    def figure(
        self,
        text: str,
        labels: list[str] | None = None,
        *,
        flags: list[str] | None = None,
        **where: object,
    ) -> Block:
        cleaned, text_flags = self._clean(text)
        clean_labels: list[str] = []
        for label in labels or []:
            cleaned_label, count = clean_text(label)
            self.invisible_removed += count
            if cleaned_label:
                clean_labels.append(" ".join(cleaned_label.split()))
        body = "\n".join(part for part in [cleaned, *clean_labels] if part)
        figure_flags, figure_ceiling = self._assess(body, check_math=False)
        block = Block(
            type="figure",
            text=body,
            labels=clean_labels,
            location=self._location(**where),
            confidence=min(0.8, figure_ceiling),
            flags=[*(flags or []), *text_flags, *figure_flags],
        )
        self.blocks.append(block)
        return block

    # --- finishing ------------------------------------------------------------------------

    def document_flags(self) -> list[str]:
        flags: list[str] = []
        if self.invisible_removed:
            flags.append(f"invisible_characters_removed:{self.invisible_removed}")
        if self.garbled_blocks:
            flags.append(f"garbled_glyphs_in_blocks:{self.garbled_blocks}")
        if self.math_blocks:
            flags.append(f"math_notation_blocks:{self.math_blocks}")
        return flags
