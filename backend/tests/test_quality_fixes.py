"""Tests for the extraction-quality fixes found on a real lecture PDF: stray glyphs from maths
fonts, mathematical notation, fake headings made of bold phrases, and formulas mistaken for charts.

Special characters are built from code points on purpose (see test_ingest_text.py)."""

import io

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.models import Block, Location, PageReport, ParsedDocument
from backend.app.ingest.parse import build_report, decide_status
from backend.app.ingest.parse_pdf import _Element, _formula_element, _is_formula_region, _is_heading, _Line, parse_pdf
from backend.app.ingest.text import math_ratio, stray_glyphs
from backend.tests.fakes import make_settings

SINHALA_HAT = chr(0x0DE1)  # how one maths font stores the "hat" accent
MALAYALAM_BAR = chr(0x0D25)  # ... and the "bar" accent
ETHIOPIC = chr(0x1218)
PRIVATE_USE = chr(0xE000)
THETA, MATH_ITALIC_THETA = chr(0x0398), chr(0x1D703)
MU, SIGMA = chr(0x03BC), chr(0x03C3)
SUM = chr(0x2211)

# --- stray glyphs -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Ordinary English text with numbers 1,234 and symbols % $ (x).",
        f"Greek is real text: {MU} and {SIGMA} and {THETA}",
        "Cyrillic is real text: " + "".join(chr(c) for c in range(0x0410, 0x0420)),
        "Japanese is real text: " + "".join(chr(c) for c in range(0x3042, 0x3052)),
        "Arabic is real text: " + "".join(chr(c) for c in range(0x0627, 0x0637)),
    ],
    ids=["empty", "ascii", "greek", "cyrillic", "japanese", "arabic"],
)
def test_real_text_has_no_stray_glyphs(text):
    assert stray_glyphs(text) == 0


@pytest.mark.parametrize(
    ("text", "count"),
    [
        (f"The estimator {THETA}{SINHALA_HAT} is unbiased", 1),
        (f"a sample mean x{MALAYALAM_BAR} and variance", 1),
        (f"value {SINHALA_HAT}{MALAYALAM_BAR}{ETHIOPIC} here in a longer sentence of ordinary text", 3),
        (f"private use {PRIVATE_USE} glyph in a normal English sentence", 1),
    ],
    ids=["sinhala", "malayalam", "three-scripts", "private-use"],
)
def test_a_few_foreign_characters_inside_ordinary_text_are_flagged(text, count):
    assert stray_glyphs(text) == count


def test_a_whole_passage_in_one_of_those_scripts_is_real_writing_not_junk():
    sinhala_sentence = "".join(chr(c) for c in range(0x0D85, 0x0D9B))  # 22 Sinhala letters

    assert stray_glyphs(sinhala_sentence) == 0


# --- mathematical notation --------------------------------------------------------------


def test_prose_is_not_math():
    assert math_ratio("Revenue grew by 12% in 2024 compared with 2023.") == 0.0
    assert math_ratio("") == 0.0 and math_ratio("   ") == 0.0


def test_the_occasional_greek_letter_in_a_sentence_stays_below_the_threshold():
    assert math_ratio(f"The sample has mean {MU} and variance {SIGMA} squared, as expected here.") < 0.2


def test_an_extracted_formula_is_high():
    formula = f"E {THETA} = {MATH_ITALIC_THETA} {SINHALA_HAT}"

    assert math_ratio(formula) >= 0.5
    assert math_ratio(f"{SUM} X = {MU}") >= 0.5


# --- the block builder flags and lowers confidence -----------------------------------------


def test_a_paragraph_with_a_stray_glyph_is_flagged_and_loses_confidence():
    builder = BlockBuilder()

    block = builder.paragraph(f"The estimator {THETA}{SINHALA_HAT} is unbiased and efficient.")

    assert block is not None and block.flags == ["garbled_glyphs:1"] and block.confidence == 0.6
    assert builder.document_flags() == ["garbled_glyphs_in_blocks:1"]


def test_a_formula_paragraph_is_flagged_as_math_notation():
    builder = BlockBuilder()

    block = builder.paragraph(f"E {THETA} = {MATH_ITALIC_THETA}")

    assert block is not None and "math_notation" in block.flags and block.confidence == 0.7
    assert builder.document_flags() == ["math_notation_blocks:1"]


def test_a_clean_paragraph_is_untouched():
    builder = BlockBuilder()

    block = builder.paragraph("Revenue grew strongly in every region.")

    assert block is not None and block.flags == [] and block.confidence == 1.0
    assert builder.document_flags() == []


def test_an_explicit_confidence_and_flags_are_kept_and_the_lower_ceiling_wins():
    builder = BlockBuilder()

    marked = builder.paragraph("a region of maths", confidence=0.7, flags=["math_region"])
    garbled = builder.paragraph(f"x{SINHALA_HAT} in a longer ordinary sentence here", confidence=0.9)

    assert marked is not None and marked.flags == ["math_region"] and marked.confidence == 0.7
    assert garbled is not None and garbled.confidence == 0.6  # the stray glyph caps it below the 0.9 given


def test_a_heading_with_a_stray_glyph_is_flagged_but_not_judged_as_maths():
    builder = BlockBuilder()

    block = builder.heading(f"The estimator {THETA}{SINHALA_HAT}", 2)

    assert block is not None and block.flags == ["garbled_glyphs:1"] and block.confidence == 0.6


def test_a_table_cell_with_a_stray_glyph_flags_the_whole_table():
    builder = BlockBuilder()

    block = builder.table([["Name", "Value"], [f"x{MALAYALAM_BAR}", "5"]])

    assert block is not None and "garbled_glyphs:1" in block.flags and block.confidence == 0.6


def test_a_figure_with_a_stray_glyph_in_its_labels_is_flagged():
    builder = BlockBuilder()

    block = builder.figure("Figure 1", [f"mean {SINHALA_HAT}"])

    assert "garbled_glyphs:1" in block.flags and block.confidence == 0.6


# --- the document status says so plainly -------------------------------------------------


def parsed_with(blocks):
    return ParsedDocument(kind="pdf", parser="t", blocks=blocks, pages=[PageReport(index=1)])


def a_block(flags=None, confidence=1.0, kind="paragraph"):
    return Block(type=kind, text="text", location=Location(page=1), flags=flags or [], confidence=confidence)


def test_maths_blocks_make_the_document_needs_review_with_a_specific_reason():
    blocks = [a_block(), a_block(["math_notation"], 0.7), a_block(["garbled_glyphs:2"], 0.6), a_block(["math_region"], 0.7)]

    status, detail = decide_status(parsed_with(blocks))

    assert status == "needs_review"
    assert detail == "3 block(s) contain mathematical notation or garbled symbols that did not extract reliably"


def test_maths_blocks_are_not_double_counted_as_low_confidence():
    blocks = [a_block(["math_notation"], 0.7), a_block(["ragged_rows"], 0.7, "table")]

    _, detail = decide_status(parsed_with(blocks))

    assert detail is not None
    assert "1 block(s) contain mathematical notation" in detail
    assert "1 block(s) are low confidence" in detail


def test_the_report_counts_maths_blocks_separately():
    report = build_report(parsed_with([a_block(["math_notation"], 0.7), a_block(["garbled_glyphs:1"], 0.6), a_block(confidence=0.5, kind="table")]))

    assert report["math_blocks"] == 2 and report["suspect_blocks"] == 1


# --- headings -----------------------------------------------------------------------------


def line(text, *, size=10.0, bold=False, top=100.0, height=12.0, column=0):
    return _Line(text=text, x0=72, x1=300, top=top, bottom=top + height, size=size, bold=bold, column=column)


BODY = 10.0


def test_big_text_is_a_heading_and_big_maths_is_not():
    assert _is_heading(line("Results", size=14), BODY) is True
    assert _is_heading(line(f"{SUM} X = {MU} {THETA} {MATH_ITALIC_THETA}", size=18), BODY) is False


def test_a_standalone_bold_capitalised_line_is_a_heading():
    above = line("Some earlier sentence that ended properly.", top=60)

    assert _is_heading(line("Key Results", bold=True, top=100), BODY, above) is True
    assert _is_heading(line("Key Results", bold=True, top=100), BODY, None) is True


@pytest.mark.parametrize(
    "text",
    [
        "riance unbiased estimator (MVUE)",  # the wrapped end of a sentence
        "and its parameters are fully known",
        "df=2) n, n = 6",
        MATH_ITALIC_THETA + " is a universal threshold",
    ],
)
def test_a_bold_phrase_starting_lower_case_is_never_a_heading(text):
    assert _is_heading(line(text, bold=True), BODY, None) is False


def test_bold_text_that_continues_an_unfinished_sentence_is_not_a_heading():
    above = line("We want the estimator with the minimum", top=100)

    assert _is_heading(line("Variance Unbiased Estimator", bold=True, top=112.5), BODY, above) is False


def test_bold_text_below_a_finished_sentence_or_after_a_gap_is_a_heading():
    finished = line("The estimator is unbiased.", top=100)
    unfinished = line("We want the estimator with the minimum", top=100)

    assert _is_heading(line("Variance Unbiased", bold=True, top=112.5), BODY, finished) is True
    assert _is_heading(line("Variance Unbiased", bold=True, top=140), BODY, unfinished) is True  # a visible gap


@pytest.mark.parametrize("text", ["Ends with a full stop.", "Ends with a comma,", "x" * 130, "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"])
def test_long_or_sentence_like_bold_lines_are_not_headings(text):
    assert _is_heading(line(text, bold=True), BODY, None) is False


def test_a_heading_can_open_with_a_quote_or_bracket():
    assert _is_heading(line("“Average” versus median", bold=True), BODY, None) is True
    assert _is_heading(line("(Appendix) Notes", bold=True), BODY, None) is True


def two_pdf_lines(draw) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    draw(pdf)
    pdf.save()
    return buffer.getvalue()


def parse(tmp_path, data):
    path = tmp_path / "doc.pdf"
    path.write_bytes(data)
    return parse_pdf(path, make_settings())


def test_a_bold_phrase_wrapped_onto_its_own_line_does_not_split_the_paragraph(tmp_path):
    def draw(pdf):
        pdf.setFont("Helvetica", 10)
        pdf.drawString(72, 700, "We want the estimator with the minimum")
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(72, 688, "variance unbiased estimator (MVUE)")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(72, 676, "among all unbiased estimators of the parameter in the model.")

    doc = parse(tmp_path, two_pdf_lines(draw))

    assert [b.type for b in doc.blocks] == ["paragraph"]
    assert "minimum variance unbiased estimator (MVUE) among all" in doc.blocks[0].text


def test_a_real_bold_heading_is_still_a_heading_and_the_text_below_it_a_paragraph(tmp_path):
    def draw(pdf):
        pdf.setFont("Helvetica", 10)
        pdf.drawString(72, 740, "This paragraph introduces the topic and ends properly.")
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(72, 700, "Key Results")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(72, 686, "Revenue grew strongly in every region this year, as the table shows.")

    doc = parse(tmp_path, two_pdf_lines(draw))

    assert [(b.type, b.text[:11]) for b in doc.blocks] == [("paragraph", "This paragra"), ("heading", "Key Results"), ("paragraph", "Revenue gro")]


# --- formulas that look like charts -------------------------------------------------------


def test_text_that_is_mostly_maths_marks_a_drawn_region_as_a_formula():
    assert _is_formula_region([f"E {THETA} = {MATH_ITALIC_THETA}", f"{SUM} X"]) is True
    assert _is_formula_region([f"x{SINHALA_HAT}", f"y{MALAYALAM_BAR}"]) is True  # stray glyphs alone are enough


def test_real_chart_labels_are_not_a_formula():
    assert _is_formula_region(["Revenue by year", "Fiscal year", "USD billions", "2019", "2020", "2024"]) is False
    assert _is_formula_region([]) is False


def test_a_formula_region_becomes_one_flagged_paragraph_element():
    figure = _Element(kind="figure", page=3, top=100, bottom=200, x0=72, subtype="vector", labels=["N", "1", f"E {THETA}"], page_height=842, bbox=(72, 100, 300, 200), flags=["chart_data_not_extracted"])

    formula = _formula_element(figure, 3)

    assert formula.kind == "paragraph" and formula.flags == ["math_region"] and formula.confidence == 0.7
    assert formula.text == f"N 1 E {THETA}" and formula.bbox == figure.bbox and formula.page == 3
