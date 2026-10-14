"""Turn parsed blocks into parent sections and child chunks. Pure code: no database, no model.

    blocks -> group by heading -> chunks inside each section -> parents (size-capped windows)

The rules, which backend/tests/test_chunking.py checks one by one:

 1. Every heading starts a new section (the parent). A heading is NOT part of any chunk's text;
    it lives in `section_path` and in the parent's span.
 2. A chunk never crosses a section boundary.
 3. Tables, figures and prose are never mixed in one chunk. A table is its own chunk.
 4. Adjacent small prose blocks (paragraph, list_item) merge, joined by a blank line, while the
    result stays within `target_tokens`. A merged chunk's text is exactly the canonical text
    from its first block's start to its last block's end.
 5. A block bigger than `max_tokens` is split: at sentence ends first, packing whole sentences;
    a sentence bigger than that at word boundaries; a word bigger than that at the limit.
    Pieces are exact slices of the block text, so offsets stay exact. Pieces of a split block
    never merge with neighbouring blocks.
 6. An introducing paragraph (ending in ':') stays with the first item after it, even over
    `target_tokens` (never over `max_tokens`).
 7. A table over `max_tokens` is split by rows, header rows repeated in every piece. A row is
    never cut; a row bigger than the limit is a piece of its own. A table without structured
    cells is split between lines. A piece's span covers its own rows (the first piece also the
    header); a repeated header is not part of the span.
 8. A section over `parent_max_tokens` becomes several Sections with the same path. The first
    starts at the heading, a later one at its first chunk.
 8b. Minimums (off when 0). A prose chunk under `min_tokens` joins the next block even past the
    target (up to the max), or folds into the chunk before it when they are neighbours. A parent
    under `parent_min_tokens` merges with the next parent (the last one with the previous), up to
    `parent_max_tokens`; it keeps the path of its first section. Tables and figures are exempt.
 9. `search_text` = one context line (document, section path, table units), a newline, `text`.
10. A chunk's confidence is the lowest of its blocks, its flags are theirs combined and sorted.
    A block with no text is skipped AND ends the current chunk, so a chunk's text always stays
    an exact slice of the canonical text. Offsets are those of the full block list.
11. The section path is the parser's `location.section_path`, never recomputed.

Offsets are into `canonical_text(blocks)` (every block's text joined by a blank line).
"""

import re
from dataclasses import dataclass, field

from backend.app.chunking.models import (
    Chunk,
    ChunkKind,
    ChunkLimits,
    ChunkSet,
    DocContext,
    Section,
)
from backend.app.context import estimate_tokens
from backend.app.ingest.models import Block, assign_offsets, canonical_text

_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")
_CHARS_PER_TOKEN = 3  # the same estimate estimate_tokens uses


# --- cutting one long text into exact slices ---------------------------------------------------


def split_spans(text: str, max_tokens: int) -> list[tuple[int, int]]:
    """(start, end) slices of `text`, each within `max_tokens` where it can be.

    Whole sentences are packed together; a sentence that is itself too big falls back to words,
    and a single word that is too big is cut at the limit."""

    def fits(start: int, end: int) -> bool:
        return estimate_tokens(text[start:end]) <= max_tokens

    sentences: list[tuple[int, int]] = []
    position = 0
    for gap in _SENTENCE_GAP.finditer(text):
        sentences.append((position, gap.start()))
        position = gap.end()
    sentences.append((position, len(text.rstrip())))

    units: list[tuple[int, int]] = []
    for start, end in (s for s in sentences if s[1] > s[0]):
        if fits(start, end):
            units.append((start, end))
            continue
        for word in _WORD.finditer(text, start, end):
            word_start, word_end = word.span()
            step = max_tokens * _CHARS_PER_TOKEN
            while word_end - word_start > step:
                units.append((word_start, word_start + step))
                word_start += step
            units.append((word_start, word_end))

    packed: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for start, end in units:
        if current is None:
            current = (start, end)
        elif fits(current[0], end):
            current = (current[0], end)
        else:
            packed.append(current)
            current = (start, end)
    if current is not None:
        packed.append(current)
    return packed


# --- one document's working state -------------------------------------------------------------


@dataclass
class _Section:
    path: list[str]
    heading: int | None  # index of the heading block, None for text before any heading
    items: list[int] = field(default_factory=list)


@dataclass
class _Window:
    """One future parent: a run of chunks plus the span of text it covers."""

    path: list[str]
    start: int
    end: int
    chunks: list[Chunk]
    pages: list[int]
    page_ends: list[int]

    @property
    def tokens(self) -> int:
        return sum(c.token_count for c in self.chunks)

    def absorb(self, later: _Window) -> None:
        self.end = later.end
        self.chunks += later.chunks
        self.pages += later.pages
        self.page_ends += later.page_ends


class _Builder:
    def __init__(
        self, blocks: list[Block], context: DocContext, limits: ChunkLimits
    ) -> None:
        self.blocks = blocks
        self.limits = limits
        self.canon = canonical_text(blocks)
        self.spans = assign_offsets(blocks)
        self.label = context.label + (f" ({context.entity})" if context.entity else "")

    # --- making a chunk ---

    def make(
        self,
        kind: ChunkKind,
        text: str,
        start: int,
        end: int,
        seqs: list[int],
        path: list[str],
        units: str | None = None,
    ) -> Chunk:
        used = [self.blocks[i] for i in seqs]
        pages = [b.location.page for b in used if b.location.page is not None]
        page_ends = [
            b.location.page_end or b.location.page for b in used if b.location.page
        ]
        first = used[0].location
        line = " > ".join([self.label, *path]) + (f" [{units}]" if units else "")
        return Chunk(
            ordinal=0,  # numbered when the chunks are placed in parents
            parent_ordinal=0,
            kind=kind,
            text=text,
            search_text=f"{line}\n{text}",
            section_path=list(path),
            block_seqs=sorted(seqs),
            char_start=start,
            char_end=end,
            token_count=estimate_tokens(text),
            page_start=min(pages) if pages else None,
            page_end=max(p for p in page_ends if p is not None) if page_ends else None,
            sheet=first.sheet,
            slide=first.slide,
            cell_range=first.cell_range,
            units=units,
            confidence=min(b.confidence for b in used),
            flags=sorted({f for b in used for f in b.flags}),
        )

    # --- sections ---

    def sections(self) -> list[_Section]:
        found: list[_Section] = []
        for i, block in enumerate(self.blocks):
            if block.type == "heading":
                found.append(_Section(block.location.section_path, i))
            elif not found:
                found.append(_Section(block.location.section_path, None, [i]))
            else:
                found[-1].items.append(i)
        return found

    def chunks_for(self, section: _Section) -> list[Chunk]:
        chunks: list[Chunk] = []
        pending: list[int] = []  # prose blocks waiting to be merged into one chunk
        last_whole: list[
            int
        ] = []  # blocks of the last chunk, if it was made of whole blocks

        def flush() -> None:
            nonlocal last_whole
            if not pending:
                return
            blocks = list(pending)
            if self.joins_backwards(last_whole, pending):
                chunks.pop()  # too small to stand alone: fold it into the chunk before it
                blocks = [*last_whole, *pending]
            start, end = self.spans[blocks[0]][0], self.spans[blocks[-1]][1]
            chunks.append(
                self.make(
                    "prose", self.canon[start:end], start, end, blocks, section.path
                )
            )
            last_whole = blocks
            pending.clear()

        for i in section.items:
            block = self.blocks[i]
            if not block.text.strip():
                flush()
            elif block.type == "table":
                flush()
                chunks.extend(self.table_chunks(i, section.path))
            elif block.type == "figure":
                flush()
                chunks.extend(self.pieces(i, "figure", section.path))
            elif estimate_tokens(block.text) > self.limits.max_tokens:
                flush()
                chunks.extend(self.pieces(i, "prose", section.path))
            elif not pending or self.joins(pending, i):
                pending.append(i)
            else:
                flush()
                pending.append(i)
        flush()
        return chunks

    def tokens_between(self, first: int, last: int) -> int:
        """Tokens of the canonical text from block `first` through block `last`."""
        return estimate_tokens(self.canon[self.spans[first][0] : self.spans[last][1]])

    def joins(self, pending: list[int], i: int) -> bool:
        """May block i join the chunk being built? Within the target; or an intro and its list;
        or the chunk is still under the minimum and the result stays within the max."""
        joined = self.tokens_between(pending[0], i)
        intro = self.blocks[pending[-1]].text.rstrip().endswith(":")
        too_small = (
            self.tokens_between(pending[0], pending[-1]) < self.limits.min_tokens
        )
        return (
            joined <= self.limits.target_tokens
            or (intro and joined <= self.limits.max_tokens)
            or (too_small and joined <= self.limits.max_tokens)
        )

    def joins_backwards(self, previous: list[int], pending: list[int]) -> bool:
        """A chunk under the minimum folds into the one before it when the two are made of
        neighbouring blocks (nothing between them, so the text stays an exact slice) and the
        result stays within the max. Pieces of a split block never qualify: they are not
        recorded as whole blocks."""
        if not previous or previous[-1] + 1 != pending[0]:
            return False
        if self.tokens_between(pending[0], pending[-1]) >= self.limits.min_tokens:
            return False
        return self.tokens_between(previous[0], pending[-1]) <= self.limits.max_tokens

    def pieces(self, i: int, kind: ChunkKind, path: list[str]) -> list[Chunk]:
        """A figure or prose block, whole if it fits, else cut into exact slices."""
        block, offset = self.blocks[i], self.spans[i][0]
        if estimate_tokens(block.text) <= self.limits.max_tokens:
            return [self.make(kind, block.text, *self.spans[i], [i], path)]
        return [
            self.make(kind, block.text[s:e], offset + s, offset + e, [i], path)
            for s, e in split_spans(block.text, self.limits.max_tokens)
        ]

    # --- tables ---

    def table_chunks(self, i: int, path: list[str]) -> list[Chunk]:
        block, offset = self.blocks[i], self.spans[i][0]
        data = block.table
        rows = [" | ".join(r) for r in data.rows] if data else block.text.split("\n")
        header = data.header_rows if data else 0
        units = data.units if data else None
        if estimate_tokens(block.text) <= self.limits.max_tokens or header >= len(rows):
            return [self.make("table", block.text, *self.spans[i], [i], path, units)]

        row_spans: list[tuple[int, int]] = []
        position = 0
        for row in rows:
            row_spans.append((offset + position, offset + position + len(row)))
            position += len(row) + 1  # the newline between rows
        head = "\n".join(rows[:header])

        def text_of(indexes: list[int]) -> str:
            body = "\n".join(rows[j] for j in indexes)
            return f"{head}\n{body}" if head else body

        chunks: list[Chunk] = []
        group: list[int] = []

        def emit() -> None:
            start = row_spans[group[0]][0] if chunks else row_spans[0][0]
            chunks.append(
                self.make(
                    "table",
                    text_of(group),
                    start,
                    row_spans[group[-1]][1],
                    [i],
                    path,
                    units,
                )
            )

        for j in range(header, len(rows)):
            if group and estimate_tokens(text_of([*group, j])) > self.limits.max_tokens:
                emit()
                group = []
            group.append(j)
        if group:
            emit()
        return chunks

    # --- parents ---

    def windows(self, section: _Section, chunks: list[Chunk]) -> list[_Window]:
        """Cut a section's chunks into windows of at most `parent_max_tokens`. The first window
        starts at the heading; a later one at its first chunk."""
        groups: list[list[Chunk]] = [[]]
        for chunk in chunks:
            used = sum(c.token_count for c in groups[-1])
            if groups[-1] and used + chunk.token_count > self.limits.parent_max_tokens:
                groups.append([])
            groups[-1].append(chunk)

        found: list[_Window] = []
        for number, group in enumerate(groups):
            heading = section.heading if number == 0 else None
            if heading is not None:
                start = self.spans[heading][0]
            else:
                start = group[0].char_start if group else 0
            if group:
                end = group[-1].char_end
            else:  # a heading with nothing under it
                end = self.spans[section.heading or 0][1]
            pages = [c.page_start for c in group if c.page_start is not None]
            page_ends = [c.page_end for c in group if c.page_end is not None]
            if heading is not None and self.blocks[heading].location.page:
                pages.append(self.blocks[heading].location.page or 0)
                page_ends.append(self.blocks[heading].location.page or 0)
            found.append(
                _Window(list(section.path), start, end, group, pages, page_ends)
            )
        return found

    def merge_small(self, windows: list[_Window]) -> list[_Window]:
        """Fold a window under `parent_min_tokens` into the next one (or, if it is the last,
        into the one before) while the result stays within `parent_max_tokens`. The merged
        parent keeps the path of its FIRST window; each chunk keeps its own `section_path`."""
        floor, ceiling = self.limits.parent_min_tokens, self.limits.parent_max_tokens
        if floor <= 0:
            return windows
        merged: list[_Window] = []
        for window in windows:
            if (
                merged
                and merged[-1].tokens < floor
                and merged[-1].tokens + window.tokens <= ceiling
            ):
                merged[-1].absorb(window)
            else:
                merged.append(window)
        if (
            len(merged) > 1
            and merged[-1].tokens < floor
            and merged[-2].tokens + merged[-1].tokens <= ceiling
        ):
            merged[-2].absorb(merged.pop())
        return merged

    def emit(self, windows: list[_Window], out: ChunkSet) -> None:
        """Number the parents and chunks, in reading order."""
        for window in windows:
            parent = Section(
                ordinal=len(out.parents),
                path=window.path,
                char_start=window.start,
                char_end=window.end,
                page_start=min(window.pages) if window.pages else None,
                page_end=max(window.page_ends) if window.page_ends else None,
                token_count=window.tokens,
            )
            out.parents.append(parent)
            for chunk in window.chunks:
                chunk.parent_ordinal = parent.ordinal
                chunk.ordinal = len(out.chunks)
                out.chunks.append(chunk)


def build_chunks(
    blocks: list[Block], context: DocContext, limits: ChunkLimits | None = None
) -> ChunkSet:
    """Blocks in, parents and chunks out. Pure and deterministic."""
    builder = _Builder(blocks, context, limits or ChunkLimits())
    windows: list[_Window] = []
    for section in builder.sections():
        windows += builder.windows(section, builder.chunks_for(section))
    out = ChunkSet()
    builder.emit(builder.merge_small(windows), out)
    return out
