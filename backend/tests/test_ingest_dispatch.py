"""The dispatcher: which parser runs for which kind, and what a parse result means for the
document's status. Pure functions, no database."""

import pytest

from backend.app.ingest.models import Block, Location, PageReport, ParsedDocument
from backend.app.ingest.parse import build_report, decide_status, page_count, parse_file
from backend.tests import make_files as mf
from backend.tests.fakes import make_settings

SETTINGS = make_settings()


def block(text="text", confidence=1.0, flags=None, kind="paragraph") -> Block:
    return Block(
        type=kind,
        text=text,
        location=Location(page=1),
        confidence=confidence,
        flags=flags or [],
    )


def doc(blocks, pages=None, kind="pdf") -> ParsedDocument:
    return ParsedDocument(
        kind=kind,
        parser="test",
        blocks=blocks,
        pages=pages if pages is not None else [PageReport(index=1)],
    )


# --- status decisions -------------------------------------------------------------------


def test_a_clean_document_is_parsed():
    assert decide_status(doc([block(), block(kind="heading")])) == ("parsed", None)


def test_a_fully_scanned_document_needs_ocr():
    pages = [
        PageReport(index=1, status="needs_ocr"),
        PageReport(index=2, status="needs_ocr"),
    ]

    status, detail = decide_status(doc([], pages))

    assert (
        status == "needs_ocr" and detail is not None and "2 page(s) need OCR" in detail
    )


def test_some_scanned_pages_make_the_document_needs_review_and_name_the_pages():
    pages = [
        PageReport(index=1),
        PageReport(index=2, status="needs_ocr"),
        PageReport(index=5, status="needs_ocr"),
    ]

    status, detail = decide_status(doc([block()], pages))

    assert status == "needs_review" and detail is not None and "2, 5" in detail


def test_unreadable_pages_make_it_needs_review():
    status, detail = decide_status(
        doc([block()], [PageReport(index=1), PageReport(index=3, status="failed")])
    )

    assert (
        status == "needs_review"
        and detail is not None
        and "page(s) 3 could not be read" in detail
    )


@pytest.mark.parametrize(
    "suspect",
    [
        block(confidence=0.7, flags=["ragged_rows"], kind="table"),
        block(confidence=0.5, flags=["table_total_mismatch:r3c1"], kind="table"),
        block(confidence=1.0, flags=["table_total_mismatch:r2c2"], kind="table"),
    ],
    ids=["ragged", "wrong-total", "wrong-total-full-confidence"],
)
def test_a_suspect_block_makes_the_document_needs_review(suspect):
    status, detail = decide_status(doc([block(), suspect]))

    assert status == "needs_review" and detail is not None and "1 block(s)" in detail


def test_a_document_with_no_content_and_no_scans_needs_review():
    status, detail = decide_status(doc([], [PageReport(index=1, status="empty")]))

    assert (status, detail) == ("needs_review", "no readable content was found")


def test_no_content_with_failed_pages_says_which_failed():
    status, detail = decide_status(doc([], [PageReport(index=1, status="failed")]))

    assert status == "needs_review" and detail is not None and "failed" in detail


def test_several_problems_are_all_reported():
    pages = [
        PageReport(index=1, status="needs_ocr"),
        PageReport(index=2, status="failed"),
    ]

    _, detail = decide_status(doc([block(confidence=0.5, kind="table")], pages))

    assert detail is not None and detail.count(";") == 2


def test_long_page_lists_are_shortened():
    pages = [PageReport(index=i, status="needs_ocr") for i in range(1, 30)]

    _, detail = decide_status(doc([block()], pages))

    assert detail is not None and detail.count(",") <= 10 and "..." in detail


# --- report and counts ------------------------------------------------------------------


def test_the_report_summarises_pages_flags_and_block_counts():
    parsed = doc(
        [block(kind="heading"), block(), block(), block(confidence=0.5, kind="table")]
    )
    parsed.flags.append("hidden_content_removed:3")

    report = build_report(parsed)

    assert report["block_counts"] == {"heading": 1, "paragraph": 2, "table": 1}
    assert report["suspect_blocks"] == 1 and report["flags"] == [
        "hidden_content_removed:3"
    ]
    assert report["pages"][0]["index"] == 1 and report["pages"][0]["status"] == "ok"


def test_page_count_only_for_paged_kinds():
    assert (
        page_count(doc([], [PageReport(index=1), PageReport(index=2)], kind="pdf")) == 2
    )
    assert page_count(doc([], [PageReport(index=1)], kind="pptx")) == 1
    assert page_count(doc([], [PageReport(index=1)], kind="docx")) is None
    assert page_count(doc([], [PageReport(index=1)], kind="html")) is None


# --- dispatch ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "make"),
    [
        pytest.param("pdf", lambda: mf.pdf_text("A real sentence in a PDF."), id="pdf"),
        pytest.param("docx", mf.docx_bytes, id="docx"),
        pytest.param("xlsx", mf.xlsx_bytes, id="xlsx"),
        pytest.param("pptx", mf.pptx_bytes, id="pptx"),
        pytest.param(
            "html", lambda: b"<html><body><p>Hello</p></body></html>", id="html"
        ),
        pytest.param("csv", lambda: b"a,b\n1,2\n", id="csv"),
        pytest.param("txt", lambda: b"plain words here", id="txt"),
        pytest.param("md", lambda: b"# Title\n\nbody text", id="md"),
    ],
)
def test_every_kind_has_a_working_parser_and_comes_out_parsed(tmp_path, kind, make):
    path = (
        tmp_path / "stored-file"
    )  # stored files have no extension: parsers must not need one
    path.write_bytes(make())

    parsed = parse_file(path, kind, SETTINGS)

    assert parsed.kind == kind and parsed.blocks
    assert decide_status(parsed)[0] == "parsed"
