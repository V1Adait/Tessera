"""The chunking rules, one test per rule. These are the specification for build_chunks.

Sizes: limits below are tiny so tests stay readable. Tokens are estimated as ceil(chars / 3), so
    target 10 tokens = 30 characters   (merging stops here)
    max    20 tokens = 60 characters   (a hard ceiling; bigger things are split)
    parent 60 tokens = 180 characters  (a section bigger than this becomes several parents)

Blocks are built with the real BlockBuilder, so `location.section_path` is exactly what the
parser produces: a heading's own path includes the heading itself.
"""

from backend.app.chunking.models import ChunkLimits, ChunkSet, DocContext
from backend.app.chunking.splitter import build_chunks
from backend.app.context import estimate_tokens
from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import (
    Block,
    Location,
    TableData,
    assign_offsets,
    canonical_text,
    render_table,
)

LIM = ChunkLimits(target_tokens=10, max_tokens=20, parent_max_tokens=60)
CTX = DocContext(label="Acme 10-K.pdf", entity="Acme")


class Doc:
    """Builds blocks the way the parser does."""

    def __init__(self) -> None:
        self._b = BlockBuilder()

    @property
    def blocks(self) -> list[Block]:
        return self._b.blocks

    def h(self, text: str, level: int = 1) -> Doc:
        self._b.heading(text, level)
        return self

    def p(self, text: str, **where: object) -> Doc:
        self._b.paragraph(text, **where)
        return self

    def li(self, text: str) -> Doc:
        self._b.list_item(text)
        return self

    def raw(self, block: Block) -> Doc:
        self._b.blocks.append(block)
        return self

    def figure(self, text: str) -> Doc:
        location = Location(section_path=self._b.section_path)
        return self.raw(Block(type="figure", text=text, location=location))

    def table(
        self,
        rows: list[list[str]],
        *,
        header_rows: int = 1,
        units: str | None = None,
        **where: object,
    ) -> Doc:
        location = Location(section_path=self._b.section_path, **where)  # type: ignore[arg-type]
        data = TableData(rows=rows, header_rows=header_rows, units=units)
        return self.raw(
            Block(type="table", text=render_table(rows), table=data, location=location)
        )


def fixed(tag: str, length: int) -> str:
    """A one-sentence paragraph of exactly `length` characters (a single long word)."""
    return (f"{tag} " + "w" * length)[: length - 1] + "."


def run(doc: Doc, limits: ChunkLimits = LIM) -> ChunkSet:
    result = build_chunks(doc.blocks, CTX, limits)
    check_invariants(doc.blocks, result, limits)
    return result


def check_invariants(
    blocks: list[Block],
    result: ChunkSet,
    limits: ChunkLimits,
    allow_oversize: bool = False,
) -> None:
    """Things that must hold for EVERY document, whatever the content."""
    canon = canonical_text(blocks)
    spans = assign_offsets(blocks)
    assert [c.ordinal for c in result.chunks] == list(range(len(result.chunks)))
    assert [p.ordinal for p in result.parents] == list(range(len(result.parents)))
    covered: set[int] = set()
    for c in result.chunks:
        assert c.text and c.text == c.text.strip()
        assert c.token_count == estimate_tokens(c.text)
        if not allow_oversize:
            assert c.token_count <= limits.max_tokens, c.text
        parent = result.parents[c.parent_ordinal]
        assert parent.char_start <= c.char_start <= c.char_end <= parent.char_end
        assert c.block_seqs == sorted(set(c.block_seqs))
        assert all(0 <= i < len(blocks) for i in c.block_seqs)
        piece = canon[c.char_start : c.char_end]
        if c.kind == "table":  # a later piece of a table starts with a repeated header
            assert c.text.endswith(piece)
        else:
            assert (
                c.text == piece
            )  # an exact slice: a quote can be found by plain matching
        assert c.search_text.endswith("\n" + c.text)
        covered.update(range(c.char_start, c.char_end))
    for block, (start, end) in zip(blocks, spans, strict=True):
        if block.type == "heading":
            continue
        for i in range(start, end):
            assert canon[i].isspace() or i in covered, (
                f"text lost at {i}: {canon[i - 5 : i + 5]!r}"
            )
    for a, b in zip(result.parents, result.parents[1:], strict=False):
        assert a.char_end <= b.char_start


# --- the shape of the result --------------------------------------------------------------


def test_no_blocks_means_nothing_to_chunk():
    assert build_chunks([], CTX, LIM) == ChunkSet()


def test_default_limits_are_used_when_none_are_given():
    doc = Doc().h("Item 7").p("A short paragraph.")

    result = build_chunks(doc.blocks, CTX)

    assert len(result.chunks) == 1


def test_the_same_input_gives_the_same_output():
    doc = Doc().h("A").p(fixed("x", 40)).p(fixed("y", 40)).h("B").p(fixed("z", 70))

    first = build_chunks(doc.blocks, CTX, LIM)
    second = build_chunks(doc.blocks, CTX, LIM)

    assert first.model_dump() == second.model_dump()


# --- rule 1 and 2: sections ---------------------------------------------------------------


def test_every_heading_starts_a_new_section_and_chunks_never_cross_it():
    doc = Doc().h("Item 7").p("Liquidity text.").h("Item 8").p("Notes text.")

    result = run(doc)

    assert [p.path for p in result.parents] == [["Item 7"], ["Item 8"]]
    assert [(c.parent_ordinal, c.text) for c in result.chunks] == [
        (0, "Liquidity text."),
        (1, "Notes text."),
    ]


def test_a_heading_is_not_part_of_any_chunk_text_but_is_in_the_path():
    doc = Doc().h("Item 7").p("Liquidity text.")

    result = run(doc)

    assert "Item 7" not in result.chunks[0].text
    assert result.chunks[0].section_path == ["Item 7"]


def test_the_parent_span_runs_from_its_heading_to_its_last_chunk():
    doc = Doc().h("Item 7").p("Liquidity text.").h("Item 8").p("Notes text.")
    spans = assign_offsets(doc.blocks)

    result = run(doc)

    assert result.parents[0].char_start == spans[0][0]
    assert result.parents[0].char_end == spans[1][1]
    assert result.parents[1].char_start == spans[2][0]
    assert result.parents[1].char_end == spans[3][1]


def test_nested_headings_give_a_path_and_a_sibling_resets_it():
    doc = (
        Doc()
        .h("Item 7", 1)
        .h("Liquidity", 2)
        .p("About cash.")
        .h("Capital", 2)
        .p("About capex.")
        .h("Item 8", 1)
        .p("About notes.")
    )

    result = run(doc)

    paths = [c.section_path for c in result.chunks]
    assert paths == [["Item 7", "Liquidity"], ["Item 7", "Capital"], ["Item 8"]]


def test_text_before_the_first_heading_belongs_to_a_root_section():
    doc = Doc().p("Preamble text.").h("Item 1").p("Body text.")

    result = run(doc)

    assert result.parents[0].path == []
    assert result.chunks[0].parent_ordinal == 0
    assert result.chunks[1].parent_ordinal == 1


def test_a_document_without_headings_is_one_root_section():
    result = run(Doc().p("One.").p("Two."))

    assert len(result.parents) == 1 and result.parents[0].path == []


def test_a_heading_with_nothing_under_it_is_a_section_without_chunks():
    doc = Doc().h("Empty").h("Next").p("Some text.")

    result = run(doc)

    assert [p.path for p in result.parents] == [["Empty"], ["Next"]]
    assert [c.parent_ordinal for c in result.chunks] == [1]
    assert result.parents[0].token_count == 0


# --- rule 4: merging small blocks ---------------------------------------------------------


def test_small_blocks_merge_until_the_target_is_reached():
    doc = Doc().p("One two.").p("Three four.").p("Five six seven.")

    result = run(doc)

    assert [c.text for c in result.chunks] == [
        "One two.\n\nThree four.",  # 21 characters, within the 30 target
        "Five six seven.",  # adding it would reach 38
    ]
    assert result.chunks[0].block_seqs == [0, 1]
    assert result.chunks[1].block_seqs == [2]


def test_a_block_over_the_target_but_within_the_max_stays_whole_and_alone():
    doc = Doc().p(fixed("a", 40)).p(fixed("b", 40))

    result = run(doc)

    assert [len(c.text) for c in result.chunks] == [40, 40]


def test_block_seqs_index_into_the_input_list():
    doc = Doc().h("Item 7").p("One.").p("Two.")

    result = run(doc)

    assert result.chunks[0].block_seqs == [1, 2]


def test_a_whitespace_only_block_is_skipped_and_ends_the_chunk():
    doc = Doc().p("Before it.").raw(Block(type="paragraph", text="   ")).p("After it.")

    result = run(doc)

    assert [c.text for c in result.chunks] == ["Before it.", "After it."]
    spans = assign_offsets(doc.blocks)
    assert result.chunks[1].char_start == spans[2][0]  # offsets count the skipped block


# --- rule 5: splitting a block that is too big --------------------------------------------

A = "Alpha beta gamma delta."  # 23
B = "Epsilon zeta eta theta."  # 23
C = "Iota kappa lambda mu."  # 21


def test_a_big_paragraph_is_split_at_sentence_ends():
    doc = Doc().p(f"{A} {B} {C}")  # 69 characters: over the 60 max

    result = run(doc)

    assert [c.text for c in result.chunks] == [f"{A} {B}", C]


def test_split_pieces_are_exact_slices_of_the_original():
    doc = Doc().p(f"{A} {B} {C}")
    canon = canonical_text(doc.blocks)

    result = run(doc)

    for c in result.chunks:
        assert canon[c.char_start : c.char_end] == c.text
    assert {tuple(c.block_seqs) for c in result.chunks} == {(0,)}


def test_a_sentence_over_the_max_is_split_between_words():
    sentence = (
        " ".join(["abcdefgh"] * 12) + "."
    )  # 12 words, no inner sentence end, 108 chars
    doc = Doc().p(sentence)

    result = run(doc)

    assert len(result.chunks) >= 2
    words = [w for c in result.chunks for w in c.text.rstrip(".").split()]
    assert words == ["abcdefgh"] * 12  # no word cut, none lost


def test_a_single_word_over_the_max_is_cut_at_the_limit():
    doc = Doc().p("x" * 130)

    result = run(doc)

    assert len(result.chunks) >= 3
    assert "".join(c.text for c in result.chunks) == "x" * 130


def test_a_big_block_does_not_merge_with_its_neighbours():
    doc = Doc().p("Tiny.").p(f"{A} {B} {C}").p("Small.")

    result = run(doc)

    assert [c.text for c in result.chunks] == ["Tiny.", f"{A} {B}", C, "Small."]


# --- rule 3: one kind of content per chunk ------------------------------------------------

ROWS = [["Item", "FY24", "FY23"], ["Revenue", "100", "90"], ["Costs", "60", "55"]]


def test_a_table_is_its_own_chunk_between_prose():
    doc = Doc().p("Before.").table(ROWS).p("After.")

    result = run(doc)

    assert [(c.kind, c.text) for c in result.chunks] == [
        ("prose", "Before."),
        ("table", render_table(ROWS)),
        ("prose", "After."),
    ]


def test_a_figure_is_its_own_chunk():
    doc = Doc().p("Before.").figure("Chart: revenue by year").p("After.")

    result = run(doc)

    assert [c.kind for c in result.chunks] == ["prose", "figure", "prose"]


def test_paragraphs_and_list_items_share_a_prose_chunk():
    result = run(Doc().p("Intro text").li("- one").li("- two"))

    assert [c.kind for c in result.chunks] == ["prose"]


# --- rule 6: an introduction stays with its list ------------------------------------------


def test_an_intro_ending_with_a_colon_stays_with_the_first_item():
    doc = (
        Doc()
        .p("Earlier text goes here.")  # 23
        .p("The covenants are:")  # 18
        .li(
            "Leverage below 3.5x"
        )  # 19: intro + item = 39, over the 30 target, within max
        .li("Cover above 2.0x")  # 16
    )

    result = run(doc)

    assert [c.text for c in result.chunks] == [
        "Earlier text goes here.",
        "The covenants are:\n\nLeverage below 3.5x",
        "Cover above 2.0x",
    ]


# --- rule 7: splitting a big table by rows ------------------------------------------------

BIG = [
    ["Item", "FY24", "FY23"],  # 18 characters per line
    ["Revenue", "100", "90"],
    ["Costs", "60", "55"],
    ["Margin", "40", "35"],
    ["Tax", "10", "9"],
    ["Profit", "30", "26"],
]
HEADER = "Item | FY24 | FY23"


def test_a_big_table_is_split_by_rows_and_every_piece_repeats_the_header():
    doc = Doc().table(BIG, units="in millions")  # 100 characters: over the 60 max

    result = run(doc)

    assert len(result.chunks) == 3
    for c in result.chunks:
        assert c.kind == "table"
        assert c.text.split("\n")[0] == HEADER
    data_rows = [line for c in result.chunks for line in c.text.split("\n")[1:]]
    assert data_rows == [
        " | ".join(r) for r in BIG[1:]
    ]  # each row once, in order, uncut


def test_table_pieces_cover_their_own_rows_in_the_original_text():
    doc = Doc().table(BIG)
    canon = canonical_text(doc.blocks)

    result = run(doc)

    first, second, _ = result.chunks
    assert (
        first.char_start == 0 and canon[first.char_start : first.char_end] == first.text
    )
    assert canon[second.char_start : second.char_end] == "\n".join(
        second.text.split("\n")[1:]
    )


def test_every_piece_of_a_split_table_carries_the_units_in_its_search_text():
    result = run(Doc().table(BIG, units="in millions"))

    assert all("in millions" in c.search_text.split("\n")[0] for c in result.chunks)
    assert all(c.units == "in millions" for c in result.chunks)


def test_two_header_rows_are_both_repeated():
    rows = [["Label", "FY24", "FY23"], ["Item", "USD", "USD"]] + BIG[1:]
    doc = Doc().table(rows, header_rows=2)

    result = run(doc)

    assert len(result.chunks) >= 2
    for c in result.chunks:
        assert c.text.split("\n")[:2] == ["Label | FY24 | FY23", "Item | USD | USD"]


def test_a_row_bigger_than_the_limit_becomes_a_piece_of_its_own_and_is_never_cut():
    giant = ["x" * 100, "y", "z"]
    rows = [["A", "B", "C"], ["short", "1", "2"], giant, ["tail", "3", "4"]]
    doc = Doc().table(rows)

    result = build_chunks(doc.blocks, CTX, LIM)
    check_invariants(doc.blocks, result, LIM, allow_oversize=True)

    assert len(result.chunks) == 3
    middle = result.chunks[1]
    assert middle.text == "A | B | C\n" + " | ".join(giant)
    assert middle.token_count > LIM.max_tokens


def test_a_table_without_structured_cells_is_split_between_lines():
    lines = "\n".join(
        f"row {i} | value {i}" for i in range(12)
    )  # 12 short lines, 195 characters
    table = Block(type="table", text=lines, table=None)
    doc = Doc().raw(table)

    result = run(doc)

    assert len(result.chunks) >= 2
    assert "\n".join(c.text for c in result.chunks) == lines


def test_a_small_table_stays_whole():
    result = run(Doc().table(ROWS))

    assert len(result.chunks) == 1


# --- rule 8: parents have a size cap ------------------------------------------------------


def test_a_huge_section_becomes_several_parents_with_the_same_path():
    doc = Doc().h("Item 7")
    for i in range(6):
        doc.p(
            fixed(f"p{i}", 40)
        )  # each is a 14-token chunk; 4 fit in a 60-token parent

    result = run(doc)

    assert len(result.chunks) == 6
    assert [p.path for p in result.parents] == [["Item 7"], ["Item 7"]]
    assert [c.parent_ordinal for c in result.chunks] == [0, 0, 0, 0, 1, 1]
    assert all(p.token_count <= LIM.parent_max_tokens for p in result.parents)
    assert result.parents[0].token_count == 4 * 14


def test_the_first_window_starts_at_the_heading_and_later_ones_at_their_first_chunk():
    doc = Doc().h("Item 7")
    for i in range(6):
        doc.p(fixed(f"p{i}", 40))
    spans = assign_offsets(doc.blocks)

    result = run(doc)

    assert result.parents[0].char_start == spans[0][0]
    assert result.parents[1].char_start == result.chunks[4].char_start


def test_a_parent_token_count_is_the_sum_of_its_chunks():
    doc = Doc().h("A").p("One.").p("Two.")

    result = run(doc)

    assert result.parents[0].token_count == sum(c.token_count for c in result.chunks)


# --- rule 9: the search text --------------------------------------------------------------


def test_search_text_is_a_context_line_then_the_exact_text():
    doc = Doc().h("Item 7", 1).h("Liquidity", 2).p("Cash was strong.")

    chunk = run(doc).chunks[0]

    line, _, rest = chunk.search_text.partition("\n")
    assert rest == "Cash was strong."
    for part in ("Acme 10-K.pdf", "Acme", "Item 7", "Liquidity"):
        assert part in line
    assert line.index("Item 7") < line.index("Liquidity")
    assert chunk.text == "Cash was strong."  # the citation text carries no context line


def test_the_context_line_works_without_a_section_or_an_entity():
    doc = Doc().p("Plain text.")

    result = build_chunks(doc.blocks, DocContext(label="notes.txt"), LIM)

    line = result.chunks[0].search_text.split("\n")[0]
    assert "notes.txt" in line


def test_the_same_text_under_different_headings_hashes_differently():
    doc = Doc().h("Risk").p("Same words.").h("Outlook").p("Same words.")

    result = run(doc)

    assert result.chunks[0].text == result.chunks[1].text
    assert result.chunks[0].content_hash != result.chunks[1].content_hash


def test_the_hash_is_stable():
    doc = Doc().h("Risk").p("Same words.")

    assert run(doc).chunks[0].content_hash == run(doc).chunks[0].content_hash


# --- rule 10: confidence, flags, location -------------------------------------------------


def test_a_chunk_takes_the_lowest_confidence_and_all_flags_of_its_blocks():
    doc = Doc()
    doc.raw(Block(type="paragraph", text="First.", confidence=0.9, flags=["z_flag"]))
    doc.raw(Block(type="paragraph", text="Second.", confidence=0.6, flags=["a_flag"]))

    chunk = run(doc).chunks[0]

    assert chunk.confidence == 0.6
    assert chunk.flags == ["a_flag", "z_flag"]


def test_pages_come_from_the_blocks_and_a_table_can_span_pages():
    doc = (
        Doc()
        .p("On three.", page=3)
        .p("On four.", page=4)
        .table(ROWS, page=5, page_end=6)
    )

    prose, table = run(doc).chunks

    assert (prose.page_start, prose.page_end) == (3, 4)
    assert (table.page_start, table.page_end) == (5, 6)


def test_sheet_slide_and_cell_range_are_kept():
    doc = Doc().table(ROWS, sheet="Data", cell_range="A1:C3")

    chunk = run(doc).chunks[0]

    assert (chunk.sheet, chunk.cell_range) == ("Data", "A1:C3")


def test_a_parent_page_range_covers_its_chunks():
    doc = (
        Doc()
        .h(
            "A",
        )
        .p("x.", page=2)
        .p(fixed("y", 40), page=3)
    )

    result = run(doc)

    assert (result.parents[0].page_start, result.parents[0].page_end) == (2, 3)
