"""The question classes and the words that define them. One source of truth: the app and the
eval harness (eval/router) both import these, so a result in one is a result about the other.

Changing any wording below changes what the classifiers do. Bump CLASSES_VERSION when you do,
and re-run eval/router/run_eval.py.
"""

from typing import Literal

CLASSES_VERSION = "2026-10-13.1"

QuestionType = Literal[
    "lookup",
    "multi_field",
    "comparison",
    "trend",
    "aggregation",
    "evaluative",
    "chat",
]

CLASSES: dict[str, str] = {
    "lookup": "One fact for one company and one period, e.g. a single figure, date or name.",
    "multi_field": "Several named facts about one company, asked together.",
    "comparison": "Compares two or more companies, or parts of one company, on the same metric.",
    "trend": "How one metric changed across several periods, for one company.",
    "aggregation": "Which or how many companies in a set meet a condition.",
    "evaluative": "Open judgement: good investment, risky, healthy, or an overview with no named facts.",
    "chat": "Greeting, thanks, or something that is not a question about documents.",
}

DECOMPOSE_HINT = (
    "Does answering need several separate lookups combined, not one passage?"
)

# What the router does when it cannot trust a label: the most general path, where an LLM
# planner decides what to look up.
SAFE_TYPE: QuestionType = "evaluative"
