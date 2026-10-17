"""What intake knows about a file: who it is about, and what was decided."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")  # same rule as the entities table
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Labels start and end with a letter or digit; the last label (the TLD) has at least two characters.
# The same rule is checked by the entities table (migration 0014).
DOMAIN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9][a-z0-9-]*[a-z0-9]$"
)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://")


def normalise_domain(value: object) -> str | None:
    """A website as a bare domain: "https://www.Palantir.com/investors" -> "palantir.com".

    Anything that is not a plausible domain is None, never a guess."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    text = _SCHEME.sub("", text)
    text = re.split(r"[/?#:\s]", text, maxsplit=1)[0]
    text = text.removeprefix("www.").strip(".")
    return text if len(text) <= 100 and DOMAIN.match(text) else None


class Identity(BaseModel):
    """The model's reading of a file's first page. Unchecked claims until `identify` has validated
    them: the company must actually appear in the text."""

    company: str | None = Field(default=None, max_length=200)
    ticker: str | None = None
    doc_type: str | None = Field(default=None, max_length=40)
    period: str | None = Field(default=None, max_length=40)
    domain: str | None = None  # the company's website, only if the page prints it
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("company", "doc_type", "period", mode="before")
    @classmethod
    def _clean_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return None
        value = " ".join(_CONTROL.sub(" ", value).split())
        return value or None

    @field_validator("domain", mode="before")
    @classmethod
    def _clean_domain(cls, value: object) -> object:
        return normalise_domain(value)

    @field_validator("ticker", mode="before")
    @classmethod
    def _clean_ticker(cls, value: object) -> object:
        if not isinstance(value, str):
            return None
        value = value.strip().upper()
        return value if TICKER.match(value) else None


class IntakeOutcome(BaseModel):
    """What intake did with one file."""

    outcome: Literal["assigned", "created", "needs_review", "skipped"]
    entity_id: str | None = None
    detail: str | None = None
    candidates: list[str] = Field(default_factory=list)
    detected: Identity | None = None
