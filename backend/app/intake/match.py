"""Which row does a file belong to? Pure rules: no database, no model.

The model only says who a file is ABOUT. Whether that is an existing row, a new one, or a question
for a person is decided here, in code that can be tested. The rules lean towards asking: a file in
the wrong row quietly poisons that row's answers, while a question costs one click.

    ticker equal            -> that row
    same name (any case, ignoring "Inc." and punctuation, or a stored alias) -> that row
    one name starts the other ("Palantir" / "Palantir Technologies") -> that row, if the extra
        words are generic ("Technologies", "Holdings"); otherwise a question ("Apple" / "Apple
        Hospitality REIT" must never merge)
    several rows fit        -> a question
    same name, different ticker -> a question
    nothing fits            -> a new row
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from backend.app.db.records import EntityRecord
from backend.app.intake.models import Identity

_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "plc", "llc", "lp", "llp", "sa", "ag", "nv", "se", "bv", "gmbh",
}  # fmt: skip

# Words that add no identity: "Palantir Technologies" is "Palantir". Anything else after a shared
# start ("Apple" -> "Apple Hospitality") is a different company until a person says otherwise.
_GENERIC = {
    "technologies", "technology", "holdings", "holding", "group", "international",
    "intl", "global", "systems", "platforms", "industries", "enterprises", "worldwide",
}  # fmt: skip

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _tokens(name: str) -> list[str]:
    tokens = _NON_WORD.sub(" ", name.lower()).split()
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return tokens


def normalise(name: str) -> str:
    return " ".join(_tokens(name))


def display_name(name: str) -> str:
    """The name for a NEW row: the company as written, without a trailing legal suffix."""
    words = name.strip().rstrip(".,").split()
    while len(words) > 1 and re.sub(r"[^a-z]", "", words[-1].lower()) in _SUFFIXES:
        words.pop()
        words[-1] = words[-1].rstrip(",")
    return " ".join(words)


@dataclass(frozen=True)
class Match:
    how: Literal["ticker", "exact", "prefix"]
    entity: EntityRecord
    extra: tuple[str, ...] = ()  # for a prefix match: the words the longer name adds


def _names(entity: EntityRecord) -> list[list[str]]:
    return [_tokens(n) for n in [entity.name, *entity.aliases] if _tokens(n)]


def find_matches(identity: Identity, entities: list[EntityRecord]) -> list[Match]:
    if identity.ticker:
        by_ticker = [e for e in entities if e.ticker == identity.ticker]
        if by_ticker:
            return [Match("ticker", e) for e in by_ticker]
    if not identity.company:
        return []
    mine = _tokens(identity.company)
    if not mine:
        return []
    exact = [e for e in entities if any(mine == theirs for theirs in _names(e))]
    if exact:
        return [Match("exact", e) for e in exact]
    found: list[Match] = []
    for e in entities:
        for theirs in _names(e):
            shorter, longer = sorted((mine, theirs), key=len)
            if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
                found.append(Match("prefix", e, tuple(longer[len(shorter) :])))
                break
    return found


@dataclass(frozen=True)
class Decision:
    action: Literal["assign", "create", "review"]
    reason: str
    entity: EntityRecord | None = None
    new_name: str | None = None
    new_ticker: str | None = None
    learn_alias: str | None = None
    learn_ticker: str | None = None
    learn_domain: str | None = None
    new_domain: str | None = None
    candidates: list[str] = field(default_factory=list)


def _review(reason: str, candidates: list[str] | None = None) -> Decision:
    return Decision("review", reason, candidates=candidates or [])


def decide(
    identity: Identity, entities: list[EntityRecord], *, min_confidence: float
) -> Decision:
    if not identity.company:
        return _review("could not tell which company this file is about")
    if identity.confidence < min_confidence:
        return _review(
            f"not sure which company this is ('{identity.company}', "
            f"confidence {identity.confidence:.2f})"
        )
    matches = find_matches(identity, entities)
    if not matches:
        return Decision(
            "create",
            f"no row for '{identity.company}' yet",
            new_name=display_name(identity.company),
            new_ticker=identity.ticker,
            new_domain=identity.domain,
        )
    if len(matches) > 1:
        names = [m.entity.name for m in matches]
        return _review(f"more than one row could be '{identity.company}'", names)

    only = matches[0]
    entity = only.entity
    if (
        only.how != "ticker"
        and entity.ticker
        and identity.ticker
        and entity.ticker != identity.ticker
    ):
        return _review(
            f"'{identity.company}' looks like the row '{entity.name}' but its ticker is "
            f"{identity.ticker}, not {entity.ticker}",
            [entity.name],
        )
    if only.how == "prefix" and not set(only.extra) <= _GENERIC:
        return _review(
            f"'{identity.company}' looks like the row '{entity.name}' but is not clearly "
            "the same company",
            [entity.name],
        )
    known = {normalise(entity.name), *(normalise(a) for a in entity.aliases)}
    alias = identity.company if normalise(identity.company) not in known else None
    return Decision(
        "assign",
        f"matched the row '{entity.name}' by {only.how}",
        entity=entity,
        learn_alias=alias,
        learn_ticker=identity.ticker if entity.ticker is None else None,
        learn_domain=identity.domain if entity.domain is None else None,
    )
