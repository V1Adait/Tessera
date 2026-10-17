"""What a tool is, and the rules every tool call goes through.

A tool is a function the model may ask for. The model only ever sees a name, a description and a
JSON schema; it never runs anything. Our code receives the request, and decides what happens:

    unknown tool              -> an error goes back to the model (it can correct itself)
    arguments fail the schema -> the validation message goes back to the model
    `gate` says it needs a yes (it costs money, or cannot be undone)
                              -> NOT run. Stored as a pending action; a person confirms in the UI
    otherwise                 -> run, with a timeout; the result goes back to the model

Nothing here knows about HTTP, the chat loop or the database tables: those are in loop.py,
actions.py and tools_impl.py.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

import asyncpg
from pydantic import BaseModel, Field, ValidationError

from backend.app.config import Settings
from backend.app.documents_service import DocumentService
from backend.app.embeddings.indexer import Indexer
from backend.app.matrix.runs import RunManager
from backend.app.matrix.service import MatrixService
from backend.app.planning.planner import ColumnPlanner

logger = logging.getLogger(__name__)

Risk = Literal["costly", "destructive"]


@dataclass
class Services:
    """The things tools act through. Any may be missing (no model key, no vector index): the tool
    that needs it then says so, instead of the whole agent failing."""

    matrix: MatrixService | None = None
    runs: RunManager | None = None
    documents: DocumentService | None = None
    indexer: Indexer | None = None
    planner: ColumnPlanner | None = None  # turns a question into proposed columns


@dataclass
class Source:
    """A passage the agent was shown in this turn, so a citation like [S2] can be checked."""

    label: str
    document_id: UUID
    filename: str
    page: int | None
    text: str


@dataclass
class ToolContext:
    pool: asyncpg.Pool
    settings: Settings
    services: Services
    workspace_id: UUID
    conversation_id: UUID
    sources: dict[str, Source] = field(
        default_factory=dict
    )  # label -> passage, this turn
    # True only when a PERSON confirmed the stored call (ActionDesk sets it). A tool that does
    # something it showed in a confirmation card refuses to run without it, so a model that passes
    # the same arguments itself cannot skip the card.
    confirmed: bool = False

    def next_label(self) -> str:
        return f"S{len(self.sources) + 1}"


class ToolResult(BaseModel):
    ok: bool
    text: str  # what the MODEL sees
    summary: str = ""  # a short line for the person
    changed_matrix: bool = False
    data: dict[str, Any] = Field(
        default_factory=dict
    )  # for the screen, e.g. the run's id


class Gate(BaseModel):
    """A tool call that must not run until a person says yes."""

    risk: Risk
    summary: str  # exactly what will happen, in plain words
    # What the confirmation card shows beyond the summary (for a column plan: the columns).
    plan: dict[str, Any] | None = None
    # Replaces the model's arguments in the stored action, so that what runs on a yes is exactly
    # what the card showed, not what the model typed.
    args: dict[str, Any] | None = None


Handler = Callable[[ToolContext, Any], Awaitable[ToolResult]]
Gater = Callable[[ToolContext, Any], Awaitable[Gate | None]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    label: str  # shown while it runs: "Searching the documents"
    description: str  # the model reads this to decide when to call it
    args: type[BaseModel]
    run: Handler
    gate: Gater | None = None  # None: never needs a yes


def fail(text: str) -> ToolResult:
    return ToolResult(ok=False, text=text, summary=text[:120])


def cap_text(text: str, limit: int) -> str:
    """A tool result is bounded: a huge answer would eat the model's whole context window."""
    if len(text) <= limit:
        return text
    return text[: limit - 40].rstrip() + "\n…[cut: the result was too long]"


class Registry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("two tools share a name")
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """The tool list in the OpenAI function format, which LangChain's bind_tools accepts for
        every provider."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": _clean_schema(t.args.model_json_schema()),
                },
            }
            for t in self._tools.values()
        ]


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic adds a 'title' to every object and field; the model gains nothing from them."""
    cleaned = {k: v for k, v in schema.items() if k != "title"}
    props = cleaned.get("properties")
    if isinstance(props, dict):
        # A property marked {"hidden": true} is for stored actions only: the model is not told it exists.
        cleaned["properties"] = {
            name: {k: v for k, v in spec.items() if k not in ("title", "hidden")}
            for name, spec in props.items()
            if not spec.get("hidden")
        }
    if "$defs" in cleaned and "$ref" not in json.dumps(cleaned.get("properties", {})):
        del cleaned["$defs"]  # the hidden property was the only user of them
    return cleaned


class Invalid(Exception):
    """The model's arguments do not fit the tool's schema."""


def parse_args(spec: ToolSpec, raw: Any) -> BaseModel:
    """Validate what the model sent. Raises Invalid with a message the model can act on."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            raise Invalid("the arguments were not valid JSON") from None
    if not isinstance(raw, dict):
        raise Invalid("the arguments must be a JSON object")
    try:
        return spec.args.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
            for e in exc.errors()[:4]
        )
        raise Invalid(problems) from None


async def run_tool(
    spec: ToolSpec, ctx: ToolContext, args: BaseModel, timeout_s: float
) -> ToolResult:
    """Run a tool with a time limit. A crash or timeout becomes a failed result, never an
    exception: one broken tool must not end the conversation."""
    try:
        result = await asyncio.wait_for(spec.run(ctx, args), timeout_s)
    except TimeoutError:
        return fail(f"{spec.name} took longer than {timeout_s:.0f} s and was stopped.")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("tool %s crashed", spec.name)
        return fail(f"{spec.name} failed unexpectedly. Nothing was changed.")
    return result.model_copy(
        update={"text": cap_text(result.text, ctx.settings.agent_tool_result_chars)}
    )
