"""Cleaning text that comes out of documents.

Two jobs: make text consistent (so search and quote matching work), and defend against text
that is invisible to a human but readable by a model. Zero-width characters, bidirectional
overrides and Unicode "tag" characters can smuggle instructions or spoof what a person sees,
so they are removed, and the number removed is reported so the document can be flagged.
"""

import re
import unicodedata

# characters a reader cannot see: zero-width, soft hyphen, word joiner, BOM, bidi controls
_INVISIBLE = {
    0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E,
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069,
    0x200E, 0x200F, 0x061C,
}  # fmt: skip
# Unicode "tag" block (U+E0000..U+E007F): used to hide ASCII text from humans
_TAG_RANGE = range(0xE0000, 0xE0080)

_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
# unusual spaces (no-break, en/em, thin, ideographic ...), built from code points so that no
# invisible character ever appears literally in this file
_SPACE_CODES = (0x00A0, 0x1680, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000)
_SPACES = re.compile("[ 	" + "".join(map(chr, _SPACE_CODES)) + "]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _is_control(ch: str) -> bool:
    return unicodedata.category(ch) == "Cc" and ch not in "\n\t"


def clean_text(text: str) -> tuple[str, int]:
    """Return (clean text, number of invisible or control characters removed)."""
    text = unicodedata.normalize("NFC", text)
    removed = 0
    kept: list[str] = []
    for ch in text:
        code = ord(ch)
        if code in _INVISIBLE or code in _TAG_RANGE or _is_control(ch):
            removed += 1
            continue
        kept.append(ch)
    text = "".join(kept)
    for ligature, plain in _LIGATURES.items():
        text = text.replace(ligature, plain)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text).strip()
    return text, removed


def clean_cell(text: object) -> str:
    """A table cell: one line, cleaned."""
    cleaned, _ = clean_text("" if text is None else str(text))
    return " ".join(cleaned.split())


def join_lines(lines: list[str]) -> str:
    """Join the lines of one paragraph. A word split by a hyphen at a line end is repaired
    ('inter-' + 'national' -> 'international') only when the next line continues in lower case,
    so real hyphenated words and list dashes are left alone."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif (
            out.endswith("-")
            and len(out) > 1
            and out[-2].isalpha()
            and line[:1].islower()
        ):
            out = out[:-1] + line
        else:
            out += " " + line
    return out


def flag_for_removed(count: int) -> list[str]:
    """Flags to attach to a block when invisible characters were stripped from it."""
    return [f"invisible_characters_removed:{count}"] if count else []
