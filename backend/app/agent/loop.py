"""The agent loop: the model asks for tools, our code decides, until the model answers.

    messages = system + history + the user's message
    repeat, at most `agent_max_steps` times:
        1. check the budget (time, this turn's cost, this conversation's cost)
        2. call the model with the tool list; stream its words to the user as they arrive
        3. no tool requested?  that was the answer -> done
        4. for each tool requested: validate -> maybe wait for a yes -> run -> a result message
        5. append the model's message and the results; go again

The model never runs anything. It can only ask. Every request passes through `handle_call`, which
is where unknown tools, bad arguments, confirmations and timeouts are dealt with. The loop always
ends: by an answer, by the step limit, or by a budget limit. It never raises to the caller;
trouble ends the stream with an ErrorEvent.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage

from backend.app.agent.prompts import AGENT_PROMPT_VERSION
from backend.app.agent.tools import (
    Invalid,
    Registry,
    ToolContext,
    ToolResult,
    parse_args,
    run_tool,
)
from backend.app.config import Settings
from backend.app.costs import Cost, get_price_book, reported_cost, resolve_cost
from backend.app.db import agent_repository as arepo
from backend.app.db.records import TraceRecord, TraceStatus
from backend.app.llm import classify_error
from backend.app.models import (
    ConfirmationEvent,
    Delta,
    Done,
    ErrorEvent,
    StreamEvent,
    ToolEvent,
    Usage,
)
from backend.app.tracing import Recorder

logger = logging.getLogger(__name__)


@dataclass
class Budget:
    """What this turn has used so far. Checked before every model call."""

    started: float = field(default_factory=time.perf_counter)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Decimal = Decimal(0)
    unknown_cost_calls: int = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def over_budget(
    budget: Budget, settings: Settings, session_spent: Decimal
) -> str | None:
    """Why the next model call must not happen, or None. Unknown-cost calls cannot be priced, so
    they are counted by the trace but cannot trip a dollar cap: this is a known limit."""
    if budget.elapsed() > settings.agent_turn_timeout_s:
        return "This took longer than the time limit for one message."
    if budget.cost_usd >= settings.agent_turn_cost_cap_usd:
        return f"This message reached its spending limit (${settings.agent_turn_cost_cap_usd})."
    if session_spent + budget.cost_usd >= settings.session_cost_cap_usd:
        return (
            f"This conversation reached its spending limit (${settings.session_cost_cap_usd}). "
            "Start a new chat to continue."
        )
    return None


@dataclass
class AgentDeps:
    """Everything the loop needs from outside, passed in so tests can supply fakes."""

    llm: BaseChatModel
    registry: Registry
    ctx: ToolContext
    settings: Settings
    provider: str
    model: str
    request_id: UUID
    recorder: Recorder


def _id_of(call: Mapping[str, Any]) -> str:
    return str(call.get("id") or uuid4())


async def _record_model_call(
    deps: AgentDeps,
    status: TraceStatus,
    message: AIMessageChunk | None,
    latency_ms: int,
    first_token_ms: int | None,
    error_code: str | None,
) -> tuple[int, int, Decimal, bool]:
    """One trace row per model call. Returns (tokens in, tokens out, cost, cost_unknown)."""
    usage = message.usage_metadata if message is not None else None
    tokens_in = usage["input_tokens"] if usage else 0
    tokens_out = usage["output_tokens"] if usage else 0
    cost = None
    if usage is not None:
        cost = resolve_cost(
            reported=reported_cost(getattr(message, "response_metadata", None)),
            book=get_price_book(),
            provider=deps.provider,
            model=deps.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    elif message is not None and (message.content or message.tool_calls):
        # Text arrived but no usage did (the stream was cut): billed, amount unknown, never free.
        cost = Cost(None, "unknown")
    trace = TraceRecord(
        request_id=uuid4(),
        conversation_id=deps.ctx.conversation_id,
        kind="agent",
        status=status,
        provider=deps.provider,
        model=deps.model,
        prompt_version=AGENT_PROMPT_VERSION,
        tokens_in=tokens_in if usage else None,
        tokens_out=tokens_out if usage else None,
        cost_usd=cost.usd if cost else None,
        cost_source=cost.source if cost else None,
        latency_ms=latency_ms,
        first_token_ms=first_token_ms,
        error_code=error_code,
    )
    await deps.recorder(trace)
    return (
        tokens_in,
        tokens_out,
        cost.usd if cost and cost.usd is not None else Decimal(0),
        cost is not None and cost.usd is None,
    )


async def _audit(
    deps: AgentDeps,
    step: int,
    name: str,
    args: Any,
    outcome: str,
    summary: str,
    started: float,
) -> None:
    """The audit line for a tool call. A failure to write it must not break the turn."""
    try:
        await arepo.record_tool_call(
            deps.ctx.pool,
            conversation_id=deps.ctx.conversation_id,
            request_id=deps.request_id,
            step=step,
            tool=name,
            args=args if isinstance(args, dict) else {"raw": str(args)[:500]},
            outcome=outcome,
            summary=summary,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
    except Exception:
        logger.exception("could not record a tool call")


async def handle_call(
    deps: AgentDeps, step: int, call: dict[str, Any]
) -> AsyncIterator[ToolEvent | ConfirmationEvent | ToolMessage]:
    """One tool request from the model. Yields events for the screen, and finally the ToolMessage
    that goes back to the model. Never raises."""
    call_id, name, raw = _id_of(call), str(call.get("name") or ""), call.get("args")
    started = time.perf_counter()
    spec = deps.registry.get(name)

    def reply(text: str) -> ToolMessage:
        return ToolMessage(content=text, tool_call_id=call_id, name=name or "unknown")

    if spec is None:
        text = f"There is no tool called '{name}'. The tools are: {', '.join(deps.registry.names())}."
        await _audit(deps, step, name, raw, "unknown_tool", text, started)
        yield ToolEvent(
            id=call_id, name=name, label="Unknown tool", status="error", summary=text
        )
        yield reply(text)
        return

    try:
        args = parse_args(spec, raw)
    except Invalid as exc:
        text = f"The arguments for {name} are not valid: {exc}. Fix them and call it again."
        await _audit(deps, step, name, raw, "invalid", text, started)
        yield ToolEvent(
            id=call_id, name=name, label=spec.label, status="error", summary=str(exc)
        )
        yield reply(text)
        return

    shown = args.model_dump(exclude_none=True)
    yield ToolEvent(id=call_id, name=name, label=spec.label, status="running")

    # A call that costs money or cannot be undone does NOT run. It is stored, with a plain summary,
    # and waits for a person.
    try:
        gate = await spec.gate(deps.ctx, args) if spec.gate else None
    except Exception:
        logger.exception("gate for %s crashed", name)
        text = f"{name} could not be checked, so it was not run. Nothing was changed."
        await _audit(deps, step, name, shown, "error", text, started)
        yield ToolEvent(
            id=call_id, name=name, label=spec.label, status="error", summary=text
        )
        yield reply(text)
        return
    if gate is not None:
        action = await arepo.create_action(
            deps.ctx.pool,
            conversation_id=deps.ctx.conversation_id,
            workspace_id=deps.ctx.workspace_id,
            request_id=deps.request_id,
            tool=name,
            args=gate.args if gate.args is not None else shown,
            risk=gate.risk,
            summary=gate.summary,
        )
        text = (
            "NOT DONE. This needs the user's confirmation first, and it is now waiting: "
            f"{gate.summary} Tell the user, in one sentence, that they must press Confirm or "
            "Cancel. Do not call this tool again."
        )
        await _audit(
            deps, step, name, shown, "needs_confirmation", gate.summary, started
        )
        yield ConfirmationEvent(
            action_id=action.id,
            tool=name,
            risk=gate.risk,
            summary=gate.summary,
            plan=gate.plan,
        )
        yield ToolEvent(
            id=call_id,
            name=name,
            label=spec.label,
            status="needs_confirmation",
            summary=gate.summary,
        )
        yield reply(text)
        return

    result: ToolResult = await run_tool(
        spec, deps.ctx, args, deps.settings.agent_tool_timeout_s
    )
    await _audit(
        deps,
        step,
        name,
        shown,
        "ok" if result.ok else "error",
        result.summary or result.text,
        started,
    )
    yield ToolEvent(
        id=call_id,
        name=name,
        label=spec.label,
        status="ok" if result.ok else "error",
        summary=result.summary,
        changed_matrix=result.changed_matrix,
    )
    yield reply(result.text)


async def run_agent(
    deps: AgentDeps, messages: list[BaseMessage], session_spent: Decimal
) -> AsyncIterator[StreamEvent]:
    """The loop. `messages` is the finished prompt (system, history, the new user message)."""
    settings = deps.settings
    budget = Budget()
    warnings: list[str] = []
    try:
        bound = deps.llm.bind_tools(deps.registry.schemas())
    except NotImplementedError:
        yield ErrorEvent(
            request_id=deps.request_id,
            code="bad_request",
            message="The chat model cannot use tools. Choose a model that supports tool calling.",
            retryable=False,
        )
        return
    conversation = list(messages)

    for step in range(1, settings.agent_max_steps + 1):
        reason = over_budget(budget, settings, session_spent)
        if reason is not None:
            yield ErrorEvent(
                request_id=deps.request_id,
                code="budget",
                message=reason,
                retryable=False,
            )
            return

        # --- one model call, streamed ------------------------------------------------------
        full: AIMessageChunk | None = None
        status: TraceStatus = "cancelled"  # stays so if the client goes away mid-call
        error_code: str | None = None
        first_token_ms: int | None = None
        call_started = time.perf_counter()
        failed: ErrorEvent | None = None
        try:
            async for chunk in bound.astream(conversation):
                text = str(chunk.text)
                if text:
                    if first_token_ms is None:
                        first_token_ms = round(
                            (time.perf_counter() - call_started) * 1000
                        )
                    yield Delta(text=text)
                if isinstance(chunk, AIMessageChunk):
                    full = chunk if full is None else full + chunk
                else:  # a model that does not stream returns one whole message
                    full = AIMessageChunk(
                        content=chunk.content,
                        tool_calls=getattr(chunk, "tool_calls", []),
                        usage_metadata=getattr(chunk, "usage_metadata", None),
                        response_metadata=getattr(chunk, "response_metadata", {}),
                    )
            status = "ok"
        except Exception as exc:
            code, message, retryable = classify_error(exc)
            logger.exception("agent model call failed", extra={"code": code})
            status, error_code = "error", code
            failed = ErrorEvent(
                request_id=deps.request_id,
                code=code,
                message=message,
                retryable=retryable,
            )
        finally:
            # Runs on success, failure, and when the client disconnects. Shielded: a cancelled
            # request must still leave its trace.
            latency = round((time.perf_counter() - call_started) * 1000)
            used = await asyncio.shield(
                _record_model_call(
                    deps, status, full, latency, first_token_ms, error_code
                )
            )
            budget.tokens_in += used[0]
            budget.tokens_out += used[1]
            budget.cost_usd += used[2]
            budget.unknown_cost_calls += int(used[3])
        if failed is not None:
            yield failed
            return
        if full is None:  # the model said nothing at all
            full = AIMessageChunk(content="")

        # --- no tool asked for: that was the answer ------------------------------------------
        calls = list(full.tool_calls)
        if not calls and not full.invalid_tool_calls:
            yield Done(
                request_id=deps.request_id,
                usage=Usage(tokens_in=budget.tokens_in, tokens_out=budget.tokens_out),
                warnings=warnings,
            )
            return

        # --- tools asked for ---------------------------------------------------------------------
        conversation.append(AIMessage(content=full.content, tool_calls=calls))
        for bad in full.invalid_tool_calls:  # the model's JSON could not even be read
            conversation.append(
                ToolMessage(
                    content="The tool call's arguments were not valid JSON. Call it again.",
                    tool_call_id=str(bad.get("id") or uuid4()),
                    name=str(bad.get("name") or "unknown"),
                )
            )
        for index, call in enumerate(calls):
            if index >= settings.agent_max_tools_per_step:
                conversation.append(
                    ToolMessage(
                        content="Too many tool calls in one step; this one was not run.",
                        tool_call_id=_id_of(call),
                        name=str(call.get("name") or "unknown"),
                    )
                )
                continue
            async for item in handle_call(deps, step, dict(call)):
                if isinstance(item, ToolMessage):
                    conversation.append(item)
                else:
                    yield item

    warnings.append(
        f"I stopped after {settings.agent_max_steps} steps before finishing. "
        "Ask me to continue if you want me to carry on."
    )
    yield Done(
        request_id=deps.request_id,
        usage=Usage(tokens_in=budget.tokens_in, tokens_out=budget.tokens_out),
        warnings=warnings,
    )
