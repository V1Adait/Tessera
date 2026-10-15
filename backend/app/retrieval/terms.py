"""Turn a natural-language question into a full-text query that can actually match.

Postgres' `websearch_to_tsquery` requires EVERY word to appear in a chunk, so a whole question
such as "What were Apple's total net sales in fiscal 2025?" matches almost nothing (in the first
retrieval eval, 13 of 25 questions returned no result at all). For recall we instead ask for ANY
of the question's content words and let the ranking put the chunks with the most of them on top.
`websearch_to_tsquery` reads the word "or" between terms as OR.
"""

import re

_WORD = re.compile(r"[A-Za-z0-9$%.,'&-]+")
_QUESTION_WORDS = frozenset(
    [
        "what",
        "was",
        "were",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "does",
        "do",
        "did",
        "how",
        "much",
        "many",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "its",
        "it",
        "as",
        "by",
        "at",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "has",
        "have",
        "had",
        "be",
        "been",
        "being",
        "can",
        "could",
        "would",
        "should",
        "will",
        "there",
        "their",
        "they",
        "them",
        "you",
        "your",
        "we",
        "our",
        "i",
        "my",
        "me",
        "about",
        "than",
        "then",
        "also",
        "into",
        "over",
        "under",
        "per",
        "if",
        "not",
        "no",
        "yes",
        "any",
        "all",
    ]
)
MAX_TERMS = 24  # a very long question is cut: more terms cost time and add little


def content_words(question: str) -> list[str]:
    """The question's words that carry meaning, lower-case, in order, without repeats."""
    words = [w.strip(".,'-&").lower() for w in _WORD.findall(question)]
    kept = [w for w in words if w and w not in _QUESTION_WORDS]
    return list(dict.fromkeys(kept))[:MAX_TERMS]


def or_query(question: str) -> str:
    """'a or b or c' for `websearch_to_tsquery`, or '' if nothing meaningful is left."""
    return " or ".join(content_words(question))
