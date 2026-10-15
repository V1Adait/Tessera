"""One way to make a small "reply with JSON" model call: invoke, trace it, parse the reply.

Used by the LLM classifier, the metric extractor and the planner, so they all get the same
tracing, error mapping and tag-escaping, and a test only has to fake one thing.
"""

import json
import logging
import re
import time
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from backend.app.costs import Cost, get_price_book, reported_cost, resolve_cost
from backend.app.db.records import TraceKind, TraceRecord
from backend.app.llm import classify_error
from backend.app.tracing import Recorder

logger = logging.getLogger(__name__)


class ModelCallError(Exception):
    """The model call failed or its reply was unusable. `code` is safe to show; the detail is
    only logged."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def neutralise(text: str, tags: tuple[str, ...]) -> str:
    """Stop user text from closing one of our data tags: '</question>' becomes '</_question>'."""
    pattern = re.compile(rf"<(/?)({'|'.join(tags)})", re.IGNORECASE)
    return pattern.sub(r"<\1_\2", text)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """The reply as a JSON object. Tolerates a markdown fence around it, nothing else."""
    try:
        value = json.loads(_FENCE.sub("", text.strip()))
    except ValueError as exc:
        raise ModelCallError("bad_output", "reply was not valid JSON") from exc
    if not isinstance(value, dict):
        raise ModelCallError("bad_output", "reply was not a JSON object")
    return value


async def ask_json(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    *,
    recorder: Recorder,
    kind: TraceKind,
    provider: str,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Call the model, write one trace row (ok or error), return the parsed JSON object."""
    started = time.perf_counter()
    tokens_in = tokens_out = None
    try:
        reply = await llm.ainvoke(messages)
        usage = reply.usage_metadata  # type: ignore[attr-defined]  # present on AIMessage
        if usage:
            tokens_in, tokens_out = usage["input_tokens"], usage["output_tokens"]
    except Exception as exc:
        code, _, _ = classify_error(exc)
        logger.exception("model call failed", extra={"kind": kind, "code": code})
        await _record(
            recorder, kind, "error", provider, model, prompt_version, started, code=code
        )
        raise ModelCallError(code, f"{kind} model call failed") from exc

    cost = resolve_cost(
        reported=reported_cost(getattr(reply, "response_metadata", None)),
        book=get_price_book(),
        provider=provider,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    await _record(
        recorder,
        kind,
        "ok",
        provider,
        model,
        prompt_version,
        started,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
    )
    return parse_json_object(str(reply.text))


async def _record(
    recorder: Recorder,
    kind: TraceKind,
    status: str,
    provider: str,
    model: str,
    prompt_version: str,
    started: float,
    *,
    code: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost: Cost | None = None,
) -> None:
    await recorder(
        TraceRecord(
            request_id=uuid4(),
            kind=kind,
            status=status,  # type: ignore[arg-type]  # "ok" | "error" only, set above
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost.usd if cost else None,
            cost_source=cost.source if cost else None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error_code=code,
        )
    )
