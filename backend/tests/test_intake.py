"""Intake: what the file really is, and what must be refused. Every refusal here is an attack or
a mistake a real upload endpoint meets."""

import io
import zipfile
from pathlib import Path

import pytest

from backend.app.ingest.intake import (
    IntakeError,
    IntakeLimits,
    detect_kind,
    safe_display_name,
)
from backend.app.ingest.storage import path_for, save_upload
from backend.tests import make_files as mf

LIMITS = IntakeLimits(
    max_bytes=5 * 1024 * 1024,
    max_zip_entries=500,
    max_zip_uncompressed_bytes=50 * 1024 * 1024,
    max_zip_ratio=100,
)


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = (
        tmp_path / "upload.bin"
    )  # the on-disk name never matters; the display name does
    path.write_bytes(data)
    return path


def kind_of(
    tmp_path: Path, display_name: str, data: bytes, limits: IntakeLimits = LIMITS
) -> str:
    return detect_kind(write(tmp_path, display_name, data), display_name, limits)


def refused(
    tmp_path: Path, display_name: str, data: bytes, limits: IntakeLimits = LIMITS
) -> IntakeError:
    with pytest.raises(IntakeError) as caught:
        kind_of(tmp_path, display_name, data, limits)
    return caught.value


# --- the accepted formats ---------------------------------------------------------------


def test_each_supported_format_is_recognised_from_its_bytes(tmp_path):
    assert kind_of(tmp_path, "report.pdf", mf.pdf_text("hi")) == "pdf"
    assert kind_of(tmp_path, "memo.docx", mf.docx_bytes()) == "docx"
    assert kind_of(tmp_path, "model.xlsx", mf.xlsx_bytes()) == "xlsx"
    assert kind_of(tmp_path, "deck.pptx", mf.pptx_bytes()) == "pptx"
    assert (
        kind_of(tmp_path, "filing.htm", b"<html><body><p>hi</p></body></html>")
        == "html"
    )
    assert kind_of(tmp_path, "data.csv", b"a,b\n1,2\n") == "csv"
    assert kind_of(tmp_path, "notes.txt", b"plain text") == "txt"
    assert kind_of(tmp_path, "readme.md", b"# Title\ntext") == "md"


def test_extension_case_does_not_matter(tmp_path):
    assert kind_of(tmp_path, "REPORT.PDF", mf.pdf_text("hi")) == "pdf"


# --- the extension must agree with the content -----------------------------------------


@pytest.mark.parametrize(
    ("name", "data"),
    [
        pytest.param("report.docx", mf.pdf_text("really a pdf"), id="pdf-named-docx"),
        pytest.param("report.pdf", mf.docx_bytes(), id="docx-named-pdf"),
        pytest.param("model.pptx", mf.xlsx_bytes(), id="xlsx-named-pptx"),
        pytest.param("noextension", mf.pdf_text("x"), id="pdf-without-extension"),
    ],
)
def test_a_wrong_extension_is_refused(tmp_path, name, data):
    error = refused(tmp_path, name, data)
    assert (error.status, error.code) == (415, "extension_mismatch")


def test_a_pdf_renamed_to_txt_is_still_a_pdf_and_is_refused(tmp_path):
    assert (
        refused(tmp_path, "innocent.txt", mf.pdf_text("x")).code == "extension_mismatch"
    )


# --- dangerous or unsupported content --------------------------------------------------


def test_macro_enabled_extensions_are_refused_before_anything_is_read(tmp_path):
    assert refused(tmp_path, "model.xlsm", mf.xlsx_bytes()).code == "macro_enabled"
    assert refused(tmp_path, "memo.docm", mf.docx_bytes()).code == "macro_enabled"


def test_a_docx_that_carries_a_macro_project_is_refused_even_with_a_clean_name(
    tmp_path,
):
    with_macro = mf.add_to_zip(
        mf.docx_bytes(), {"word/vbaProject.bin": b"fake macro project"}
    )

    error = refused(tmp_path, "innocent.docx", with_macro)

    assert (error.status, error.code) == (415, "macro_enabled")


def test_legacy_binary_office_files_are_refused_by_content_and_by_name(tmp_path):
    ole = bytes.fromhex("d0cf11e0a1b11ae1") + b"\x00" * 600
    assert refused(tmp_path, "old.doc", ole).code == "legacy_office_format"
    assert (
        refused(tmp_path, "renamed.docx", ole).code == "legacy_office_format"
    )  # name does not save it


def test_an_encrypted_pdf_is_refused(tmp_path):
    fake = b"%PDF-1.7\n%...\ntrailer << /Root 1 0 R /Encrypt 5 0 R >>\n%%EOF\n"
    assert refused(tmp_path, "locked.pdf", fake).code == "encrypted_pdf"


def test_binary_garbage_with_a_text_extension_is_refused(tmp_path):
    error = refused(tmp_path, "notes.txt", b"abc\x00\x01\x02binary")
    assert (error.status, error.code) == (415, "unsupported_type")


def test_unknown_extensions_are_refused(tmp_path):
    assert refused(tmp_path, "program.exe", b"just text").code == "unsupported_type"
    assert refused(tmp_path, "script.js", b"alert(1)").code == "unsupported_type"


def test_a_page_named_html_that_has_no_html_is_refused(tmp_path):
    assert refused(tmp_path, "fake.html", b"no markup at all here").code == "not_html"


def test_an_empty_file_is_refused(tmp_path):
    assert refused(tmp_path, "empty.pdf", b"").status == 400


# --- zip attacks ------------------------------------------------------------------------


def test_a_zip_bomb_is_refused_by_its_compression_ratio(tmp_path):
    bomb = mf.add_to_zip(
        mf.docx_bytes(), {"word/media/huge.bin": b"\x00" * (30 * 1024 * 1024)}
    )

    error = refused(tmp_path, "bomb.docx", bomb)

    assert (error.status, error.code) == (422, "zip_bomb")


def test_a_zip_that_would_expand_too_far_is_refused(tmp_path):
    big = mf.add_to_zip(
        mf.docx_bytes(), {f"word/part{i}.bin": b"x" * 1000 for i in range(5)}
    )
    tiny = IntakeLimits(5 * 1024 * 1024, 500, 2000, 100)

    assert refused(tmp_path, "big.docx", big, tiny).code == "zip_bomb"


def test_a_zip_with_too_many_entries_is_refused(tmp_path):
    many = mf.add_to_zip(
        mf.docx_bytes(), {f"word/extra{i}.xml": b"<a/>" for i in range(50)}
    )
    few = IntakeLimits(5 * 1024 * 1024, 20, 50 * 1024 * 1024, 100)

    assert refused(tmp_path, "many.docx", many, few).code == "zip_bomb"


# (A backslash name like "word\evil.xml" is not listed: Python's zipfile rewrites it to the
# harmless "word/evil.xml" on Windows before we can see it, so only other systems need that case.)
@pytest.mark.parametrize(
    "evil_name",
    ["../evil.xml", "word/../../evil.xml", "/etc/passwd", "C:/windows/evil.xml"],
)
def test_zip_entries_with_path_tricks_are_refused(tmp_path, evil_name):
    sneaky = mf.add_to_zip(mf.docx_bytes(), {evil_name: b"x"})

    assert refused(tmp_path, "sneaky.docx", sneaky).code == "unsafe_zip_entry"


def test_a_zip_that_is_not_an_office_document_is_refused(tmp_path):
    plain_zip = mf.zip_bytes({"readme.txt": b"hello", "data.csv": b"a,b"})
    assert refused(tmp_path, "archive.docx", plain_zip).code == "not_an_office_file"


def test_a_truncated_zip_is_refused_as_corrupt(tmp_path):
    damaged = mf.docx_bytes()[:-40]
    assert refused(tmp_path, "damaged.docx", damaged).code == "corrupt_zip"


def test_a_zip_whose_headers_lie_about_sizes_is_refused(tmp_path):
    """Declared size is tiny but the data inflates far more. Python's zipfile stops at the declared
    size and then fails the checksum, and our own streaming check is the second line of defence."""
    docx = mf.docx_bytes()
    honest = zipfile.ZipFile(io.BytesIO(docx))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as out:
        for info in honest.infolist():
            out.writestr(info, honest.read(info))
        out.writestr("word/liar.bin", b"A" * (200 * 1024))
    raw = bytearray(payload.getvalue())
    # find the central directory record of liar.bin and shrink its declared uncompressed size
    marker = b"word/liar.bin"
    index = raw.rfind(marker)
    header_start = raw.rfind(b"PK\x01\x02", 0, index)
    raw[header_start + 24 : header_start + 28] = (10).to_bytes(
        4, "little"
    )  # file_size field

    error = refused(tmp_path, "liar.docx", bytes(raw))

    assert error.status == 422
    assert error.code in {"zip_bomb", "corrupt_zip"}


# --- display names ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\secret.docx", "secret.docx"),
        ("a/b/c.xlsx", "c.xlsx"),
        ("  spaced   name .txt ", "spaced name .txt"),
        ("", "unnamed"),
        (None, "unnamed"),
        ("///", "unnamed"),
    ],
)
def test_safe_display_name(given, expected):
    assert safe_display_name(given) == expected


def test_display_names_lose_control_and_invisible_characters_and_are_capped():
    name = "bad" + chr(0) + "name" + chr(0x202E) + ".pdf"
    assert safe_display_name(name) == "badname.pdf"
    assert len(safe_display_name("x" * 1000 + ".pdf")) == 255


# --- storage ----------------------------------------------------------------------------


def reader(data: bytes):
    buffer = io.BytesIO(data)

    async def read(size: int) -> bytes:
        return buffer.read(size)

    return read


async def test_an_upload_is_stored_under_its_hash_not_its_name(tmp_path):
    data = mf.pdf_text("stored")

    stored = await save_upload(reader(data), "../../../evil name.pdf", tmp_path, LIMITS)

    assert stored.path == path_for(tmp_path, stored.sha256)
    assert stored.path.read_bytes() == data
    assert stored.kind == "pdf" and stored.size == len(data)
    assert stored.display_name == "evil name.pdf"
    assert "evil" not in str(stored.path)
    assert list((tmp_path / ".tmp").iterdir()) == []  # no temporary file left behind


async def test_the_same_bytes_are_stored_once(tmp_path):
    data = mf.docx_bytes()

    first = await save_upload(reader(data), "a.docx", tmp_path, LIMITS)
    second = await save_upload(reader(data), "copy of a.docx", tmp_path, LIMITS)

    assert first.sha256 == second.sha256 and first.path == second.path
    assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == 1


async def test_an_oversized_upload_is_refused_with_413_and_leaves_nothing(tmp_path):
    small = IntakeLimits(1000, 500, 10_000_000, 100)

    with pytest.raises(IntakeError) as caught:
        await save_upload(reader(b"x" * 5000), "big.txt", tmp_path, small)

    assert caught.value.status == 413
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


async def test_a_refused_upload_leaves_no_file_behind(tmp_path):
    with pytest.raises(IntakeError):
        await save_upload(
            reader(b"MZ\x90\x00 not a document"), "tool.exe", tmp_path, LIMITS
        )

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


async def test_an_empty_upload_is_refused(tmp_path):
    with pytest.raises(IntakeError) as caught:
        await save_upload(reader(b""), "empty.pdf", tmp_path, LIMITS)
    assert caught.value.status == 400


def test_path_for_refuses_anything_but_a_hash(tmp_path):
    with pytest.raises(ValueError):
        path_for(tmp_path, "../../etc/passwd")
    with pytest.raises(ValueError):
        path_for(tmp_path, "A" * 64)  # upper case is not a canonical hash
