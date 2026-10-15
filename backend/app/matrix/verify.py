"""Does the quote really appear in the text it cites? Plain code, no model.

This is the rule everything else in the matrix leans on: a cell is only `answered` if its evidence
survives this check. The model never sees a verdict; it is asked to quote, and the code looks.

What counts as "the same text":
  * letters, digits and punctuation must match exactly (case too: "Q1" is not "q1")
  * any run of whitespace matches any other run (a quote may break a line where the source did not)
  * in a table, a run of cell separators matches one separator: a table row is stored as
    "Total net sales |  | $416,161" and a model that writes "Total net sales | $416,161" has not
    changed what the row says
Nothing else is forgiven: not curly vs straight quotes, not "..." in the middle, not a typo. A quote
that fails is not "close enough": the cell is not answered on it.
"""

import re
from uuid import UUID

from pydantic import BaseModel

from backend.app.db.records import ChunkRecord
from backend.app.matrix.models import ProposedEvidence, VerifiedEvidence

MIN_QUOTE_CHARS = 3
MAX_QUOTE_CHARS = 1200

_SEPARATOR = r"\|(?:\s+\|)*"  # one or more table cell separators
_cache: dict[str, re.Pattern[str]] = {}


def _pattern(quote: str) -> re.Pattern[str]:
    """The regular expression that finds `quote` allowing only the whitespace and separator
    differences described above. Every literal piece is escaped, so a quote cannot inject a
    pattern of its own."""
    found = _cache.get(quote)
    if found is not None:
        return found
    pieces = [
        _SEPARATOR if token == "|" else re.escape(token) for token in quote.split()
    ]
    compiled = re.compile(r"\s+".join(pieces))
    if len(_cache) > 2000:  # a long-lived process must not grow without bound
        _cache.clear()
    _cache[quote] = compiled
    return compiled


def find_quote(quote: str, text: str) -> tuple[int, int] | None:
    """(start, end) of `quote` inside `text`, or None. The slice text[start:end] is what the
    document really says, which is what gets stored as the evidence."""
    quote = quote.strip()
    if not quote:
        return None
    match = _pattern(quote).search(text)
    return (match.start(), match.end()) if match else None


class EvidenceProblem(BaseModel):
    """Why a proposed quote was refused. `message` is written to be shown to the model on a
    retry: it says what was wrong, not who is to blame."""

    chunk: str
    message: str


def _quote_shape_problem(quote: str) -> str | None:
    stripped = quote.strip()
    if len(stripped) < MIN_QUOTE_CHARS or not any(ch.isalnum() for ch in stripped):
        return "the quote is too short to show anything: quote the words that state the answer"
    if len(stripped) > MAX_QUOTE_CHARS:
        return (
            f"the quote is longer than {MAX_QUOTE_CHARS} characters: quote only the sentence "
            "or table row that states the answer"
        )
    return None


def verify_evidence(
    proposed: list[ProposedEvidence], chunks: dict[str, ChunkRecord]
) -> tuple[list[VerifiedEvidence], list[EvidenceProblem]]:
    """Check every proposed quote against the chunk it names.

    `chunks` maps the label the model saw ("C1", "C2"...) to the chunk. Returns the evidence that
    passed and a problem for each that did not. A label that is not in `chunks` is a problem: the
    model cited a chunk it was never given. The same passage cited twice is kept once."""
    verified: list[VerifiedEvidence] = []
    problems: list[EvidenceProblem] = []
    seen: set[tuple[UUID, int, int]] = set()
    for item in proposed:
        chunk = chunks.get(item.chunk.strip().upper())
        if chunk is None:
            problems.append(
                EvidenceProblem(
                    chunk=item.chunk,
                    message=f"{item.chunk!r} is not one of the chunks you were given",
                )
            )
            continue
        shape = _quote_shape_problem(item.quote)
        if shape:
            problems.append(EvidenceProblem(chunk=item.chunk, message=shape))
            continue
        span = find_quote(item.quote, chunk.text)
        if span is None:
            problems.append(
                EvidenceProblem(
                    chunk=item.chunk,
                    message=(
                        f"the quote {item.quote.strip()[:80]!r} does not appear word for word "
                        f"in chunk {chunk_label(item.chunk)}; copy the text exactly as written"
                    ),
                )
            )
            continue
        start, end = span
        key = (chunk.id, start, end)
        if key in seen:
            continue
        seen.add(key)
        verified.append(locate(chunk, start, end))
    return verified, problems


def chunk_label(label: str) -> str:
    return label.strip().upper()


def locate(chunk: ChunkRecord, start: int, end: int) -> VerifiedEvidence:
    """Place a quote found at chunk.text[start:end] in the document's own coordinates.

    Most chunks are an exact slice of the document, so the quote's offsets are the chunk's start
    plus its position in the chunk. A table piece that repeats its header rows is NOT an exact slice
    (the repeated header is not at that place in the document), so its position inside the chunk
    does not map to the document. For those the whole chunk's span is stored and `exact_range` says
    so honestly."""
    exact = (chunk.char_end - chunk.char_start) == len(chunk.text)
    return VerifiedEvidence(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        quote=chunk.text[start:end],
        char_start=chunk.char_start + start if exact else chunk.char_start,
        char_end=chunk.char_start + end if exact else chunk.char_end,
        exact_range=exact,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_confidence=chunk.confidence,
        chunk_flags=list(chunk.flags),
    )
