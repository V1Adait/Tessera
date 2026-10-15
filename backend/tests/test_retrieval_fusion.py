"""Reciprocal Rank Fusion and the question-to-terms step: pure functions, no database."""

import pytest

from backend.app.retrieval.fusion import rrf
from backend.app.retrieval.terms import MAX_TERMS, content_words, or_query

# --- rrf --------------------------------------------------------------------------------------


def test_one_list_keeps_its_order():
    assert [i for i, _ in rrf([["a", "b", "c"]])] == ["a", "b", "c"]


def test_no_lists_or_empty_lists_give_nothing():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_the_score_is_one_over_k_plus_rank_summed_over_lists():
    fused = dict(rrf([["a", "b"], ["b", "a"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_an_item_high_in_both_lists_beats_one_first_in_only_one():
    fused = [i for i, _ in rrf([["only_first", "both", "x"], ["y", "both", "z"]])]

    assert fused[0] == "both"


def test_a_smaller_k_lets_a_first_place_count_for_more():
    lists = [["solo", "both"], ["other", "both"]]

    assert rrf(lists, k=1)[0][0] == "both"  # consensus: 1/3 + 1/3 beats 1/2
    assert rrf([["solo", "x", "both"], ["other", "y", "both"]], k=1)[0][0] == "solo"


def test_weights_scale_each_lists_say():
    lists = [["a", "b"], ["b", "a"]]

    assert rrf(lists, weights=[1.0, 1.0])[0][0] == "a"  # tie on score, a was first seen
    assert rrf(lists, weights=[1.0, 3.0])[0][0] == "b"
    assert rrf(lists, weights=[0.0, 1.0])[0][0] == "b"


def test_ties_are_broken_by_best_rank_then_by_first_seen():
    fused = [i for i, _ in rrf([["a"], ["b"]])]  # same score, same best rank

    assert fused == ["a", "b"]


def test_an_item_repeated_inside_one_list_counts_once():
    fused = dict(rrf([["a", "a", "a"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61)


def test_the_result_is_deterministic():
    lists = [["a", "b", "c", "d"], ["d", "c", "b", "a"], ["b", "d"]]

    assert rrf(lists) == rrf(lists)


def test_works_with_any_hashable_items():
    from uuid import uuid4

    ids = [uuid4() for _ in range(3)]

    assert next(i for i, _ in rrf([ids, ids[::-1]])) in ids


@pytest.mark.parametrize("k", [0, -5])
def test_k_must_be_positive(k):
    with pytest.raises(ValueError):
        rrf([["a"]], k=k)


def test_one_weight_per_list_is_required():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


# --- terms ------------------------------------------------------------------------------------


def test_question_words_are_dropped_and_content_words_kept_in_order():
    assert content_words("What were Apple's total net sales in fiscal 2025?") == [
        "apple's",
        "total",
        "net",
        "sales",
        "fiscal",
        "2025",
    ]


def test_repeats_are_removed():
    assert content_words("revenue revenue Revenue") == ["revenue"]


def test_numbers_and_money_survive():
    assert "$416,161" in content_words("Was it $416,161 or 6%?")
    assert "6%" in content_words("Was it $416,161 or 6%?")


def test_a_question_of_only_stop_words_has_no_query():
    assert or_query("what is the") == ""
    assert or_query("") == ""
    assert or_query("???") == ""


def test_or_query_joins_with_or():
    assert or_query("long term debt") == "long or term or debt"


def test_a_very_long_question_is_cut():
    question = " ".join(f"word{i}" for i in range(100))

    assert len(content_words(question)) == MAX_TERMS
