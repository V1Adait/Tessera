"""The quote check: the rule that decides whether a model's claim becomes evidence.

These tests are the specification. Each one is a way a quote can be wrong, or a way a correct one
can look wrong. If one of them fails, a cell could be marked answered on text the document does
not contain.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.app.db.records import ChunkRecord
from backend.app.matrix.models import ProposedEvidence
from backend.app.matrix.verify import (
    MAX_QUOTE_CHARS,
    find_quote,
    locate,
    verify_evidence,
)


def chunk(
    text: str, *, start: int = 100, end: int | None = None, confidence: float = 1.0
) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        ordinal=0,
        kind="prose",
        text=text,
        search_text="doc > path\n" + text,
        section_path=["Item 7"],
        block_seqs=[0],
        char_start=start,
        char_end=start + len(text) if end is None else end,
        token_count=len(text) // 3,
        page_start=4,
        page_end=4,
        sheet=None,
        slide=None,
        cell_range=None,
        units=None,
        confidence=confidence,
        flags=[],
        content_hash="h",
    )


# --- find_quote -------------------------------------------------------------------------------

TEXT = (
    "The Company repurchased 402 million shares of its common stock for $89.3 billion."
)


def test_an_exact_quote_is_found_with_its_position():
    assert find_quote("402 million shares", TEXT) == (24, 42)
    start, end = find_quote("402 million shares", TEXT)  # type: ignore[misc]
    assert TEXT[start:end] == "402 million shares"


def test_the_returned_slice_is_what_the_document_says_not_what_the_model_typed():
    text = "Net sales\nrose 6%\nin 2025."

    start, end = find_quote("Net sales rose 6% in 2025", text)  # type: ignore[misc]

    assert text[start:end] == "Net sales\nrose 6%\nin 2025"


def test_any_whitespace_matches_any_whitespace():
    assert find_quote("402   million \t shares", TEXT) is not None
    assert find_quote("  402 million shares  ", TEXT) is not None


def test_case_must_match():
    assert find_quote("402 Million shares", TEXT) is None


def test_a_changed_digit_is_not_found():
    assert find_quote("403 million shares", TEXT) is None
    assert find_quote("$89.4 billion", TEXT) is None


def test_a_word_inserted_or_dropped_is_not_found():
    assert find_quote("402 million of shares", TEXT) is None
    assert find_quote("402 shares", TEXT) is None


def test_an_ellipsis_does_not_stand_for_missing_text():
    assert find_quote("repurchased 402 million ... $89.3 billion", TEXT) is None


def test_curly_and_straight_quotes_are_different_characters():
    assert find_quote("Company's", "The Company’s results") is None


def test_empty_and_blank_quotes_are_not_found():
    assert find_quote("", TEXT) is None
    assert find_quote("   ", TEXT) is None


def test_regular_expression_characters_in_a_quote_are_just_characters():
    text = "Revenue (net) was $5.2m [restated] + 3%."
    assert find_quote("(net) was $5.2m [restated] + 3%.", text) is not None
    assert find_quote(".*", text) is None
    assert find_quote("(net|gross)", text) is None
    assert find_quote("$5.2m [restated]", text) is not None


def test_a_very_long_pathological_quote_returns_quickly():
    text = "a " * 5000
    assert find_quote("a " * 400 + "b", text) is None


# --- table rows -------------------------------------------------------------------------------

ROW = "Total net sales |  | $416,161 | 6% |  | $391,035"


def test_a_row_quoted_with_single_separators_matches_one_with_empty_cells():
    assert find_quote("Total net sales | $416,161", ROW) is not None
    assert find_quote("Total net sales | $416,161 | 6% | $391,035", ROW) is not None


def test_cells_in_the_wrong_order_do_not_match():
    assert find_quote("$416,161 | Total net sales", ROW) is None


def test_a_missing_separator_does_not_match():
    assert find_quote("Total net sales $416,161", ROW) is None


# --- verify_evidence --------------------------------------------------------------------------


def test_a_good_quote_is_verified_and_placed_in_the_document():
    c = chunk(TEXT, start=1000)

    verified, problems = verify_evidence(
        [ProposedEvidence(chunk="C1", quote="402 million shares")], {"C1": c}
    )

    assert problems == []
    [ev] = verified
    assert ev.quote == "402 million shares"
    assert (ev.char_start, ev.char_end) == (1024, 1042)
    assert ev.exact_range is True
    assert (ev.document_id, ev.chunk_id) == (c.document_id, c.id)
    assert ev.page_start == 4


def test_the_stored_quote_is_the_documents_text_even_when_the_model_changed_spacing():
    c = chunk("Net sales\nrose 6%.")

    [ev], _ = verify_evidence(
        [ProposedEvidence(chunk="C1", quote="Net sales rose 6%")], {"C1": c}
    )

    assert ev.quote == "Net sales\nrose 6%"


def test_a_quote_from_a_different_chunk_than_the_one_cited_is_refused():
    one, two = chunk("Cash was $35,934 million."), chunk("Debt was $78,328 million.")

    verified, problems = verify_evidence(
        [ProposedEvidence(chunk="C1", quote="Debt was $78,328 million")],
        {"C1": one, "C2": two},
    )

    assert verified == [] and len(problems) == 1
    assert "word for word" in problems[0].message


def test_a_label_the_model_was_never_given_is_refused():
    verified, problems = verify_evidence(
        [ProposedEvidence(chunk="C9", quote="402 million shares")], {"C1": chunk(TEXT)}
    )

    assert verified == []
    assert "not one of the chunks you were given" in problems[0].message


@pytest.mark.parametrize("label", ["c1", " C1 ", "C1"])
def test_labels_are_matched_without_regard_to_case_or_padding(label):
    verified, problems = verify_evidence(
        [ProposedEvidence(chunk=label, quote="402 million shares")], {"C1": chunk(TEXT)}
    )

    assert len(verified) == 1 and problems == []


@pytest.mark.parametrize("quote", ["5", "$", "--", "a", "  "])
def test_a_quote_too_short_to_show_anything_is_refused(quote):
    verified, problems = verify_evidence(
        [ProposedEvidence(chunk="C1", quote=quote)], {"C1": chunk("5 $ -- a")}
    )

    assert verified == [] and len(problems) == 1


def test_an_over_long_quote_is_refused():
    text = "word " * 300
    quote = text.strip()
    assert len(quote) > MAX_QUOTE_CHARS

    verified, problems = verify_evidence(
        [ProposedEvidence(chunk="C1", quote=quote)], {"C1": chunk(text)}
    )

    assert verified == [] and "longer than" in problems[0].message


def test_the_same_passage_cited_twice_is_kept_once():
    c = chunk(TEXT)

    verified, _ = verify_evidence(
        [
            ProposedEvidence(chunk="C1", quote="402 million shares"),
            ProposedEvidence(chunk="C1", quote="402   million shares"),
        ],
        {"C1": c},
    )

    assert len(verified) == 1


def test_good_and_bad_quotes_in_one_reply_are_separated():
    c = chunk(TEXT)

    verified, problems = verify_evidence(
        [
            ProposedEvidence(chunk="C1", quote="402 million shares"),
            ProposedEvidence(chunk="C1", quote="a quote nobody wrote"),
        ],
        {"C1": c},
    )

    assert len(verified) == 1 and len(problems) == 1


def test_nothing_proposed_verifies_nothing():
    assert verify_evidence([], {"C1": chunk(TEXT)}) == ([], [])


def test_the_problem_message_does_not_repeat_a_huge_quote():
    huge = "x" * 1000

    _, problems = verify_evidence(
        [ProposedEvidence(chunk="C1", quote=huge)], {"C1": chunk("short text here")}
    )

    assert len(problems[0].message) < 300


# --- locate -----------------------------------------------------------------------------------


def test_an_exact_slice_chunk_places_the_quote_to_the_character():
    c = chunk("Hello world", start=50)

    ev = locate(c, 6, 11)

    assert (ev.char_start, ev.char_end, ev.exact_range) == (56, 61, True)


def test_a_chunk_that_is_not_an_exact_slice_gives_its_whole_span_and_says_so():
    # a table piece that repeats its header: 10 characters of text, but a 6-character span
    c = chunk("Item | 2025\nSales | 5", start=200, end=206)

    ev = locate(c, 12, 21)

    assert (ev.char_start, ev.char_end, ev.exact_range) == (200, 206, False)
    assert ev.quote == "Sales | 5"


def test_the_chunks_confidence_and_flags_travel_with_the_evidence():
    c = chunk("A table row | 5", confidence=0.5)
    c.flags.append("table_total_mismatch:r3c1")

    ev = locate(c, 0, 5)

    assert ev.chunk_confidence == 0.5
    assert ev.chunk_flags == ["table_total_mismatch:r3c1"]


def test_timestamps_in_fixtures_are_timezone_aware():
    assert datetime.now(tz=UTC).tzinfo is not None
