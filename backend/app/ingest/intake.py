"""Intake: decide what an uploaded file REALLY is, and refuse anything unsafe.

Nothing the client says is trusted. The file name only labels the document for display; the type
comes from the first bytes of the file, and the extension must then agree with it. Word, Excel
and PowerPoint files are zip archives, so they are checked for zip bombs, path tricks, macros and
encryption before any parser opens them.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.app.config import Settings
from backend.app.ingest.models import DocKind
from backend.app.ingest.text import clean_text


class IntakeError(Exception):
    """An upload was refused. `status` is the HTTP status; `message` is safe to show."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class IntakeLimits:
    max_bytes: int
    max_zip_entries: int
    max_zip_uncompressed_bytes: int
    max_zip_ratio: int

    @classmethod
    def from_settings(cls, settings: Settings) -> IntakeLimits:
        return cls(
            max_bytes=settings.max_upload_bytes,
            max_zip_entries=settings.max_zip_entries,
            max_zip_uncompressed_bytes=settings.max_zip_uncompressed_bytes,
            max_zip_ratio=settings.max_zip_ratio,
        )


EXTENSIONS: dict[DocKind, tuple[str, ...]] = {
    "pdf": (".pdf",),
    "docx": (".docx",),
    "xlsx": (".xlsx",),
    "pptx": (".pptx",),
    "html": (".html", ".htm", ".xhtml"),
    "csv": (".csv",),
    "txt": (".txt",),
    "md": (".md", ".markdown"),
}
_MACRO_EXTENSIONS = (
    ".docm",
    ".dotm",
    ".xlsm",
    ".xltm",
    ".xlam",
    ".pptm",
    ".potm",
    ".ppsm",
)
_LEGACY_EXTENSIONS = (".doc", ".xls", ".ppt")
_OLE_MAGIC = bytes.fromhex(
    "d0cf11e0a1b11ae1"
)  # old binary Office formats and some encrypted ones
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
_BIG_ENTRY = (
    1024 * 1024
)  # entries smaller than this are not checked for compression ratio
_HTML_HINT = re.compile(
    rb"<\s*(!doctype\s+html|html|head|body|table|div|p|h[1-6]|span|ul|ol)\b",
    re.IGNORECASE,
)


def safe_display_name(filename: str | None) -> str:
    """The client's file name, reduced to something harmless to display and store as text.
    It is NEVER used as a path: files are stored under their hash."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name, _ = clean_text(name)
    name = " ".join(name.split())[:255].strip()
    return name or "unnamed"


def detect_kind(path: Path, display_name: str, limits: IntakeLimits) -> DocKind:
    """The document type, decided from the file's bytes. Raises IntakeError if unsupported."""
    extension = Path(display_name).suffix.lower()
    if extension in _MACRO_EXTENSIONS:
        raise IntakeError(
            415,
            "macro_enabled",
            "Macro-enabled Office files are not accepted. Save a copy without macros.",
        )
    if extension in _LEGACY_EXTENSIONS:
        raise IntakeError(
            415,
            "legacy_office_format",
            "Old binary Office files (.doc, .xls, .ppt) are not supported. Save as .docx, .xlsx or .pptx.",
        )

    with path.open("rb") as handle:
        head = handle.read(65536)
    if not head:
        raise IntakeError(400, "empty_file", "The file is empty.")

    if head.lstrip(b"\x00\r\n\t ")[:5] == b"%PDF-" or b"%PDF-" in head[:1024]:
        _reject_encrypted_pdf(path)
        return _require_extension("pdf", extension)
    if head.startswith(_OLE_MAGIC):
        raise IntakeError(
            415,
            "legacy_office_format",
            "Old binary Office files are not supported. Save as .docx, .xlsx or .pptx.",
        )
    if head.startswith(_ZIP_MAGICS):
        return _require_extension(inspect_ooxml(path, limits), extension)

    return _detect_text_kind(head, extension)


def _require_extension(kind: DocKind, extension: str) -> DocKind:
    if extension not in EXTENSIONS[kind]:
        raise IntakeError(
            415,
            "extension_mismatch",
            f"The file content is {kind.upper()} but its name ends in '{extension or '(none)'}'.",
        )
    return kind


def _reject_encrypted_pdf(path: Path) -> None:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 16384))
        tail = handle.read()
    if b"/Encrypt" in tail:
        raise IntakeError(
            422, "encrypted_pdf", "Password-protected PDFs are not supported."
        )


def _detect_text_kind(head: bytes, extension: str) -> DocKind:
    if b"\x00" in head:
        raise IntakeError(
            415,
            "unsupported_type",
            "This looks like a binary file of an unsupported type.",
        )
    if extension in EXTENSIONS["html"]:
        if not _HTML_HINT.search(head):
            raise IntakeError(
                415,
                "not_html",
                "The file is named like a web page but contains no HTML.",
            )
        return "html"
    for kind in ("csv", "txt", "md"):
        if extension in EXTENSIONS[kind]:
            return kind  # type: ignore[return-value]
    raise IntakeError(
        415,
        "unsupported_type",
        "Supported types: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), HTML, CSV, TXT, Markdown.",
    )


# --- zip based Office files -------------------------------------------------------------


def inspect_ooxml(path: Path, limits: IntakeLimits) -> DocKind:
    """Check a zip archive for bombs, unsafe names, macros and encryption, and say which Office
    format it is. Raises IntakeError."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            _check_zip_headers(infos, limits)
            names = {info.filename for info in infos}
            if "[Content_Types].xml" not in names:
                raise IntakeError(
                    415,
                    "not_an_office_file",
                    "This zip file is not a Word, Excel or PowerPoint document.",
                )
            content_types = archive.read("[Content_Types].xml")[:1_000_000].decode(
                "utf-8", "replace"
            )
            if (
                any(n.lower().endswith("vbaproject.bin") for n in names)
                or "macroEnabled" in content_types
            ):
                raise IntakeError(
                    415,
                    "macro_enabled",
                    "Documents containing macros are not accepted.",
                )
            kind = _ooxml_kind(names)
            _verify_streams(archive, infos, limits)
            return kind
    except zipfile.BadZipFile as exc:
        raise IntakeError(
            422, "corrupt_zip", "The file is a damaged zip archive."
        ) from exc


def _check_zip_headers(infos: list[zipfile.ZipInfo], limits: IntakeLimits) -> None:
    if len(infos) > limits.max_zip_entries:
        raise IntakeError(422, "zip_bomb", "The archive contains too many files.")
    if sum(info.file_size for info in infos) > limits.max_zip_uncompressed_bytes:
        raise IntakeError(
            422, "zip_bomb", "The archive would expand to an unreasonable size."
        )
    for info in infos:
        name = info.filename
        parts = PurePosixPath(name).parts
        if (
            name.startswith(("/", "\\"))
            or ".." in parts
            or "\\" in name
            or (parts and ":" in parts[0])
        ):
            raise IntakeError(
                422, "unsafe_zip_entry", "The archive contains an unsafe file path."
            )
        if info.flag_bits & 0x1:
            raise IntakeError(
                422, "encrypted_zip", "Encrypted documents are not supported."
            )
        if info.file_size > _BIG_ENTRY and info.file_size > limits.max_zip_ratio * max(
            info.compress_size, 1
        ):
            raise IntakeError(
                422,
                "zip_bomb",
                "The archive is compressed suspiciously well (possible zip bomb).",
            )


def _ooxml_kind(names: set[str]) -> DocKind:
    if "word/document.xml" in names:
        return "docx"
    if "xl/workbook.xml" in names:
        return "xlsx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    raise IntakeError(
        415,
        "not_an_office_file",
        "This zip file is not a Word, Excel or PowerPoint document.",
    )


def _verify_streams(
    archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], limits: IntakeLimits
) -> None:
    """Headers can lie about sizes, so actually decompress every entry with a hard byte cap."""
    total = 0
    for info in infos:
        if info.is_dir():
            continue
        seen = 0
        with archive.open(info) as stream:
            while chunk := stream.read(65536):
                seen += len(chunk)
                total += len(chunk)
                if (
                    seen > info.file_size + 1024
                    or total > limits.max_zip_uncompressed_bytes
                ):
                    raise IntakeError(
                        422, "zip_bomb", "The archive expands beyond its declared size."
                    )
