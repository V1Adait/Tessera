"""The rubric: the fixed menu of dimensions an open question ("is Acme a good investment?") is
broken into. It belongs to the workspace's profile (domain pack); this is the investing one.

The planner may only PICK from this list and reword the question for the company. It cannot
invent a dimension. `default_question` is what a dimension becomes when the planner fails.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    id: str
    label: str
    default_question: str  # {entity} is filled in


RUBRIC_VERSION = "investing-1"

INVESTING_RUBRIC: tuple[Dimension, ...] = (
    Dimension("business", "Business model", "What does {entity} do and how does it make money?"),
    Dimension("growth", "Growth", "How have {entity}'s revenue and earnings grown?"),
    Dimension("profitability", "Profitability", "What are {entity}'s margins and how have they moved?"),
    Dimension("leverage", "Leverage and liquidity", "How much debt does {entity} have, when does it fall due, and how much cash does it hold?"),
    Dimension("cash_flow", "Cash flow", "What cash does {entity} generate from operations and how is it used?"),
    Dimension("risks", "Risks", "What risks does {entity} state in its documents?"),
    Dimension("valuation_inputs", "Valuation inputs", "What figures in {entity}'s documents would feed a valuation: share count, EBITDA, net debt?"),
    Dimension("governance", "Management and governance", "Who runs {entity} and how is the board structured?"),
)  # fmt: skip

_BY_ID = {d.id: d for d in INVESTING_RUBRIC}


def dimension(dimension_id: str) -> Dimension | None:
    return _BY_ID.get(dimension_id)
