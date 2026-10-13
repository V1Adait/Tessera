"""Finding entities and periods in a question: pure code, so the tests are plain tables."""

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from backend.app.db.records import EntityRecord
from backend.app.query.slots import find_entities, find_periods

TODAY = date(2026, 10, 13)  # last completed year is 2025


def entity(name: str) -> EntityRecord:
    return EntityRecord(
        id=uuid4(),
        workspace_id=uuid4(),
        name=name,
        position=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# --- entities ---------------------------------------------------------------------------


def test_entities_are_found_in_the_order_they_are_mentioned():
    acme, beta = entity("Acme"), entity("Beta")

    found = find_entities("Beta vs Acme on margin", [acme, beta])

    assert [e.name for e in found] == ["Beta", "Acme"]


def test_entity_matching_ignores_case():
    assert [e.name for e in find_entities("what is acme's debt", [entity("Acme")])] == [
        "Acme"
    ]


def test_an_entity_name_inside_a_longer_word_does_not_match():
    assert find_entities("show the metadata", [entity("Meta")]) == []


def test_the_longer_name_wins_over_a_shorter_one_it_contains():
    visa, europe = entity("Visa"), entity("Visa Europe")

    found = find_entities("Visa Europe revenue", [visa, europe])

    assert [e.name for e in found] == ["Visa Europe"]


def test_a_name_with_regex_characters_is_matched_literally():
    assert find_entities("AT&T (US) debt", [entity("AT&T (US)")])
    assert find_entities("anything", [entity("a.*")]) == []


def test_an_entity_is_reported_once_even_if_mentioned_twice():
    assert len(find_entities("Acme, and again Acme", [entity("Acme")])) == 1


# --- periods ----------------------------------------------------------------------------


def labels(question: str) -> list[str]:
    return [p.label for p in find_periods(question, TODAY)]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("revenue in FY24", ["FY2024"]),
        ("revenue in FY2024", ["FY2024"]),
        ("fiscal 2023 revenue", ["FY2023"]),
        ("revenue in 2024", ["FY2024"]),
        ("compare 2022 and 2024", ["FY2022", "FY2024"]),
        ("margins 2022-2024", ["FY2022", "FY2023", "FY2024"]),
        ("margins from 2022 to 2024", ["FY2022", "FY2023", "FY2024"]),
        ("leverage since 2022", ["FY2022", "FY2023", "FY2024", "FY2025"]),
        ("growth over the last three years", ["FY2023", "FY2024", "FY2025"]),
        ("growth over the past 2 years", ["FY2024", "FY2025"]),
        ("Q1 2024 revenue", ["Q1 2024"]),
        ("2024 Q3 revenue", ["Q3 2024"]),
        ("Q2 FY24 revenue", ["Q2 2024"]),
        ("revenue", []),
    ],
)
def test_periods(question, expected):
    assert labels(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "paid $2,024 million",
        "has 2000 employees",
        "net debt of 2024.5",
        "margin of 2020%",
        "a 2025x multiple",
    ],
)
def test_numbers_that_look_like_years_are_not_periods(question):
    assert labels(question) == []


def test_periods_are_sorted_and_not_repeated():
    assert labels("2024 then 2022 then 2024 again") == ["FY2022", "FY2024"]


def test_since_a_year_in_the_future_gives_nothing():
    assert labels("since 2031") == []


def test_a_silly_range_is_ignored():
    assert labels("from 1990 to 2090") == []
