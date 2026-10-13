"""Turn parsed blocks into parent sections and child chunks. THIS IS THE PART YOU WRITE.

The tests in backend/tests/test_chunking.py are the specification. Run them with

    .venv/Scripts/python -m pytest backend/tests/test_chunking.py -q

and work down the failures. The rules, in the order the tests check them:

 1. Every heading starts a new section (a Section, the parent). A heading is NOT part of any
    chunk's text; it lives in `section_path` and in the parent's span.
 2. A chunk never crosses a section boundary.
 3. Tables, figures and prose are never mixed in one chunk. A table is its own chunk.
 4. Adjacent small prose blocks (paragraph, list_item) merge, joined by a blank line, while the
    result stays within `target_tokens`. A merged chunk's text is exactly the canonical text
    from its first block's start to its last block's end.
 5. A block bigger than `max_tokens` is split: at sentence ends first, packing whole sentences
    up to `max_tokens`; a sentence bigger than that at word boundaries; a single word bigger
    than that is cut at the limit. Pieces are exact slices of the block text (strip the
    whitespace at the cut), so offsets stay exact. Pieces of a split block never merge with
    neighbouring blocks.
 6. An introducing paragraph (one that ends with ':') stays in the same chunk as the first
    item after it, even if that goes over `target_tokens` (never over `max_tokens`).
 7. A table over `max_tokens` is split by rows, packing rows up to `max_tokens` (header
    included). Every piece repeats the header rows (`TableData.header_rows`). A row is never
    cut; a single row bigger than the limit becomes a piece on its own. A table without
    structured cells (`table is None`) is split between lines, with no header to repeat.
    A piece's char_start/char_end cover only the rows it holds from the original text (a
    repeated header is not part of that span), and the first piece also covers the header.
 8. A Section's token_count is the sum of its chunks' token_count. A section over
    `parent_max_tokens` becomes several Sections with the same path, filled in order; no chunk
    sits across two of them. The first starts at the heading, a later one at its first chunk.
 9. `search_text` = one context line, then a newline, then `text`. The line is the document
    label (and entity), then the section path, joined with ' > ', and for tables the units.
10. A chunk's confidence is the lowest of its blocks; its flags are all their flags, sorted;
    token_count is estimate_tokens(text). Pages, sheet, slide and cell_range come from the
    blocks. A block whose text is empty or only whitespace is skipped AND ends the current
    chunk (the chunk text must stay an exact slice of the canonical text). Offsets are always
    those of the full block list, skipped blocks included.
11. The section path is `location.section_path` as the parser already computed it (a heading's
    own path includes the heading itself). Do not recompute it from heading levels.

Useful pieces already there: `estimate_tokens` (backend/app/context.py), `assign_offsets` and
`render_table` (backend/app/ingest/models.py).
"""

from backend.app.chunking.models import ChunkLimits, ChunkSet, DocContext
from backend.app.ingest.models import Block


def build_chunks(
    blocks: list[Block], context: DocContext, limits: ChunkLimits | None = None
) -> ChunkSet:
    """Blocks in, parents and chunks out. Pure: no database, no model, deterministic."""
    raise NotImplementedError(
        "write me: see the rules above and backend/tests/test_chunking.py"
    )
