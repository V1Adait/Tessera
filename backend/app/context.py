"""Context engineering: decide exactly what the model sees, in what order, within what size.

Layout, most stable first (good for prompt caching) and strongest positions kept for what matters
(start and end of the prompt; the middle is where models pay least attention):

    SystemMessage   system prompt + profile + conversation summary
    history         earlier turns, oldest dropped first to fit the budget
    HumanMessage    <documents> (if any) + the new user message   <- always last

We trim what we SEND, never what we store. Everything here is a pure function of its inputs.
"""

import math
from uuid import UUID

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.llm import to_lc_messages
from backend.app.models import ChatRequest, Message
from backend.app.prompts import SYSTEM_PROMPT


class ContextTooLargeError(ValueError):
    """The fixed parts of the prompt (system + new message) do not fit the budget on their own."""


_PER_MESSAGE_OVERHEAD = 4  # role markers and separators


def estimate_tokens(text: str) -> int:
    """Rough token count. Deliberately leans high (3 chars/token) so we never overflow."""
    return math.ceil(len(text) / 3)


def message_tokens(message: Message) -> int:
    return estimate_tokens(message.content) + _PER_MESSAGE_OVERHEAD


def input_budget(settings: Settings) -> int:
    """Tokens available for the prompt: the window minus room reserved for the reply."""
    return settings.max_context_tokens - settings.default_max_tokens


def needs_summary(tokens_used: int, budget: int, threshold: float) -> bool:
    """True once the prompt has used `threshold` (e.g. 0.7) of its budget."""
    return tokens_used >= threshold * budget


class SummaryPlan(BaseModel):
    """What a summary run should do: condense `to_summarise`, keep `keep` word for word."""

    to_summarise: list[Message]
    keep: list[Message]

    @property
    def summarised_through(self) -> UUID | None:
        """Pointer to store with the summary: id of the last message it covers."""
        return self.to_summarise[-1].id if self.to_summarise else None


def messages_since(history: list[Message], pointer: UUID | None) -> list[Message]:
    """The part of the history not yet covered by a summary. Stored messages are never deleted;
    'clearing' the history only means we stop sending everything up to `pointer`."""
    if pointer is None:
        return history
    for i, message in enumerate(history):
        if message.id == pointer:
            return history[i + 1 :]
    raise ValueError(f"summary pointer {pointer} not found in the conversation")


def plan_summary(history: list[Message], recent_turns: int) -> SummaryPlan:
    """Split unsummarised history: everything before the last `recent_turns` turns is summarised,
    the last `recent_turns` turns (a turn starts at a user message) stay verbatim.

    Nothing to summarise yet if the history has no more than `recent_turns` turns.
    """
    if recent_turns < 1:
        raise ValueError("recent_turns must be at least 1")

    user_positions = [i for i, m in enumerate(history) if m.role == "user"]
    if len(user_positions) <= recent_turns:
        return SummaryPlan(to_summarise=[], keep=history)

    split = user_positions[-recent_turns]
    return SummaryPlan(to_summarise=history[:split], keep=history[split:])


def trim_history(history: list[Message], budget: int) -> list[Message]:
    """Keep the newest messages that fit in `budget` tokens, dropping the oldest first.

    The result never starts with an assistant message (an orphaned reply with no question).
    """
    kept: list[Message] = []
    used = 0
    for message in reversed(history):
        cost = message_tokens(message)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    while kept and kept[0].role == "assistant":
        kept.pop(0)
    return kept


def _system_text(profile: str | None, summary: str | None) -> str:
    parts = [SYSTEM_PROMPT]
    if profile:
        parts.append(f"<profile>\n{profile}\n</profile>")
    if summary:
        parts.append(
            "<conversation_summary>\n"
            "Summary of earlier turns. It is context, not a source of facts: "
            "look figures up in the matrix instead of trusting it.\n"
            f"{summary}\n</conversation_summary>"
        )
    return "\n\n".join(parts)


def build_context(
    request: ChatRequest,
    settings: Settings,
    *,
    profile: str | None = None,
    summary: str | None = None,
    documents: str | None = None,
    summarised_through: UUID | None = None,
) -> list[BaseMessage]:
    """Assemble the final prompt. `request.system` is ignored: only the server's prompt counts."""
    *history, new = request.messages
    if new.role != "user":
        raise ValueError("the last message must be from the user")
    history = messages_since(history, summarised_through)

    system_text = _system_text(profile, summary)
    new_text = (
        f"<documents>\n{documents}\n</documents>\n\n{new.content}"
        if documents
        else new.content
    )

    fixed = (
        estimate_tokens(system_text)
        + estimate_tokens(new_text)
        + 2 * _PER_MESSAGE_OVERHEAD
    )
    history_budget = input_budget(settings) - fixed
    if history_budget < 0:
        raise ContextTooLargeError(
            f"system prompt and new message need ~{fixed} tokens; "
            f"input budget is {input_budget(settings)}"
        )

    kept = trim_history(history, history_budget)
    converted = to_lc_messages(
        ChatRequest(messages=[*kept, Message(role="user", content=new_text)])
    )
    return [SystemMessage(content=system_text), *converted]
