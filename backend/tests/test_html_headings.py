"""Headings that are written as styled text, not as <h1>-<h6> tags (every SEC filing does this)."""

from pathlib import Path

from backend.app.config import Settings
from backend.app.ingest.parse_html import parse_html
from backend.tests.fakes import make_settings

BOLD = "font-weight:700"


def parse(tmp_path: Path, body: str, settings: Settings | None = None):
    path = tmp_path / "doc.html"
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return parse_html(path, settings or make_settings())


def kinds(doc):
    return [(b.type, b.level, b.text) for b in doc.blocks]


def line(text, style=BOLD, tag="span"):
    return f'<div><{tag} style="{style}">{text}</{tag}></div>'


def test_a_bold_item_line_is_a_level_two_heading(tmp_path):
    doc = parse(
        tmp_path, line("Item 1A.&#160;&#160;Risk Factors") + "<div>Body text.</div>"
    )

    assert kinds(doc) == [
        ("heading", 2, "Item 1A. Risk Factors"),
        ("paragraph", None, "Body text."),
    ]


def test_part_lines_are_level_one(tmp_path):
    assert kinds(parse(tmp_path, line("PART II"))) == [("heading", 1, "PART II")]


def test_item_and_part_are_recognised_even_if_not_bold(tmp_path):
    doc = parse(
        tmp_path, line("Item 7. Management Discussion", style="font-weight:400")
    )

    assert kinds(doc) == [("heading", 2, "Item 7. Management Discussion")]


def test_any_other_short_bold_line_is_a_level_three_heading(tmp_path):
    doc = parse(tmp_path, line("Liquidity and Capital Resources"))

    assert kinds(doc) == [("heading", 3, "Liquidity and Capital Resources")]


def test_the_paragraphs_below_inherit_the_section_path(tmp_path):
    doc = parse(
        tmp_path,
        line("PART II")
        + line("Item 7. Discussion")
        + line("Liquidity")
        + "<div>Cash was strong.</div>"
        + line("Item 8. Statements")
        + "<div>Notes follow.</div>",
    )

    paths = {
        b.text: b.location.section_path for b in doc.blocks if b.type == "paragraph"
    }
    assert paths["Cash was strong."] == ["PART II", "Item 7. Discussion", "Liquidity"]
    assert paths["Notes follow."] == ["PART II", "Item 8. Statements"]


def test_bold_via_a_b_tag_or_the_word_bold_counts(tmp_path):
    doc = parse(
        tmp_path,
        "<div><b>Competition</b></div>" + line("Segments", style="font-weight:bold"),
    )

    assert [b.type for b in doc.blocks] == ["heading", "heading"]


def test_regular_weight_text_stays_a_paragraph(tmp_path):
    doc = parse(tmp_path, line("Competition", style="font-weight:400"))

    assert [b.type for b in doc.blocks] == ["paragraph"]


def test_a_bold_sentence_stays_a_paragraph(tmp_path):
    doc = parse(tmp_path, line("The Company is highly competitive."))

    assert [b.type for b in doc.blocks] == ["paragraph"]


def test_a_long_bold_passage_stays_a_paragraph(tmp_path):
    long = " ".join(["Important"] * 30)

    assert [b.type for b in parse(tmp_path, line(long)).blocks] == ["paragraph"]


def test_a_bare_number_such_as_a_page_number_is_not_a_heading(tmp_path):
    assert [b.type for b in parse(tmp_path, line("12")).blocks] == ["paragraph"]


def test_a_lowercase_fragment_is_not_a_heading(tmp_path):
    assert [b.type for b in parse(tmp_path, line("continued from above")).blocks] == [
        "paragraph"
    ]


def test_only_part_of_the_line_being_bold_is_not_a_heading(tmp_path):
    body = '<div><span style="font-weight:700">Note</span><span> that costs rose</span></div>'

    assert [b.type for b in parse(tmp_path, body).blocks] == ["paragraph"]


def test_a_sentence_that_merely_mentions_an_item_is_not_a_heading(tmp_path):
    text = (
        "Item 1A of this report discusses the risks that could affect results in detail"
    )

    assert [b.type for b in parse(tmp_path, f"<div>{text}.</div>").blocks] == [
        "paragraph"
    ]


def test_hidden_bold_text_is_not_a_heading(tmp_path):
    body = line("Secret instructions", style="font-weight:700;display:none")

    assert parse(tmp_path, body).blocks == []


def test_real_heading_tags_still_work(tmp_path):
    doc = parse(tmp_path, "<h2>Real heading</h2><p>Text.</p>")

    assert kinds(doc) == [("heading", 2, "Real heading"), ("paragraph", None, "Text.")]


def test_table_header_cells_are_not_turned_into_headings(tmp_path):
    doc = parse(
        tmp_path,
        "<table><tr><th>Item</th><th>2025</th></tr><tr><td>Sales</td><td>5</td></tr></table>",
    )

    assert [b.type for b in doc.blocks] == ["table"]
