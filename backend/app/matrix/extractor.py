"""Fill one cell: find the passages, ask the model, then let the CODE decide what the answer is.

    row's documents ──> retrieve (hybrid search, scoped to this row) ──> group into parents
        ──> number the chunks C1, C2, ... and put them in the prompt's data slot
        ──> the model replies with an answer and quotes            (a CLAIM, not yet a fact)
        ──> verify each quote is in the chunk it cites               verify.py
        ──> parse the answer into a typed value                      values.py
        ──> check the value really is in the quote                   values.py
        ──> only now a status: answered / low_confidence / ...       CellOutcome

If the checks fail, the model is told exactly what was wrong and gets ONE more try. If it still
cannot back its answer, the cell is `low_confidence` with NO value, and the reason names what the
model claimed. A model's unverified claim never becomes a cell value.

Failures are explicit: a model error, an unusable reply or a retrieval failure make the cell
`pending_or_error` with a plain reason, never a silent empty cell and never a guess.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

import asyncpg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from backend.app.chunking.expand import ParentView, expand_to_parents
from backend.app.config import Settings
from backend.app.db import repository as repo
from backend.app.db.records import ChunkHit, ChunkRecord
from backend.app.matrix.models import (
    CellOutcome,
    ColumnRecord,
    ModelReply,
    VerifiedEvidence,
)
from backend.app.matrix.prompts import (
    DATA_TAGS,
    EXTRACT_PROMPT,
    EXTRACT_PROMPT_VERSION,
    answer_type_hint,
)
from backend.app.matrix.values import (
    BadValue,
    is_per_unit,
    normalise,
    supported_by_quote,
)
from backend.app.matrix.verify import verify_evidence
from backend.app.query.llm_json import ModelCallError, ask_json, neutralise
from backend.app.tracing import Recorder, no_recorder

logger = logging.getLogger(__name__)

Retriever = Callable[[UUID, str, UUID, int], Awaitable[list[ChunkHit]]]
"""(workspace_id, query, entity_id, limit) -> chunks, best first. Always scoped to the row."""

MAX_CONTEXT_CHUNKS = 60


@dataclass(frozen=True)
class LabelledChunk:
    label: str
    chunk: ChunkRecord
    filename: str


def build_context(
    parents: list[ParentView], filenames: dict[UUID, str], budget_tokens: int
) -> list[LabelledChunk]:
    """Number the chunks of the top parents (C1, C2, ...) within a token budget.

    Parents are taken best first and whole. A parent that does not fit is skipped, except that the
    best parent always contributes: if it alone is over budget, only the chunks that actually
    matched are kept, in reading order, so the prompt stays bounded."""
    chosen: list[ChunkRecord] = []
    used = 0
    for index, parent in enumerate(parents):
        cost = sum(c.token_count for c in parent.chunks)
        if used + cost <= budget_tokens:
            chosen += parent.chunks
            used += cost
        elif index == 0:
            hit_ids = set(parent.hit_chunk_ids)
            for chunk in parent.chunks:
                if chunk.id in hit_ids and used + chunk.token_count <= budget_tokens:
                    chosen.append(chunk)
                    used += chunk.token_count
        if len(chosen) >= MAX_CONTEXT_CHUNKS:
            chosen = chosen[:MAX_CONTEXT_CHUNKS]
            break
    return [
        LabelledChunk(f"C{i}", c, filenames.get(c.document_id, "document"))
        for i, c in enumerate(chosen, start=1)
    ]


def _attr(value: str) -> str:
    """Text going into a tag attribute: no quotes, no angle brackets, one line."""
    return " ".join(
        value.replace('"', "'").replace("<", "(").replace(">", ")").split()
    )[:200]


def build_messages(
    column: ColumnRecord,
    entity_name: str,
    context: list[LabelledChunk],
    problem: str | None = None,
) -> list[BaseMessage]:
    """System instructions in one slot; the question and ALL document text in the data slot, with
    anything that could close one of our tags defused."""
    safe = lambda text: neutralise(text, DATA_TAGS)
    applies = column.applicability or "applies to every company"
    chunks = "\n".join(
        f'<chunk id="{c.label}" document="{_attr(c.filename)}" '
        f'section="{_attr(" > ".join(c.chunk.section_path))}" '
        f'page="{c.chunk.page_start if c.chunk.page_start is not None else ""}">\n'
        f"{safe(c.chunk.text)}\n</chunk>"
        for c in context
    )
    body = (
        "<task>\n"
        f"<company>{safe(entity_name)}</company>\n"
        f"<column>{safe(column.name)}</column>\n"
        f"<question>{safe(column.prompt)}</question>\n"
        f"<answer_type>{answer_type_hint(column)}</answer_type>\n"
        f"<applies>{safe(applies)}</applies>\n"
        "</task>\n"
        f"<documents>\n{chunks}\n</documents>"
    )
    if problem:
        body += f"\n<problem>Your last reply was rejected: {safe(problem)} Fix it.</problem>"
    return [SystemMessage(content=EXTRACT_PROMPT), HumanMessage(content=body)]


# --- turning a reply into a decision ---------------------------------------------------------


@dataclass
class _Decision:
    outcome: CellOutcome | None = None
    problems: list[str] | None = None
    proposal: str | None = None  # what the model claimed, for the cell's explanation


def _unit_hint(
    evidence: list[VerifiedEvidence], by_id: dict[UUID, ChunkRecord]
) -> str | None:
    for item in evidence:
        chunk = by_id.get(item.chunk_id) if item.chunk_id else None
        if chunk is not None and chunk.units:
            return chunk.units
    return None


def decide(
    reply: ModelReply,
    column: ColumnRecord,
    context: list[LabelledChunk],
    *,
    low_confidence_at: float,
    final_attempt: bool,
) -> _Decision:
    """Check a model reply. Either an outcome, or the list of problems to tell the model."""
    if reply.status in ("not_found", "not_applicable"):
        return _Decision(outcome=CellOutcome(status=reply.status, detail=reply.reason))

    by_label = {c.label: c.chunk for c in context}
    by_id = {c.chunk.id: c.chunk for c in context}
    verified, evidence_problems = verify_evidence(reply.evidence, by_label)
    problems = [p.message for p in evidence_problems]

    if reply.status == "conflicting":
        if len(verified) < 2:
            problems.append(
                "conflicting needs two quotes that both appear in the chunks"
            )
            return _Decision(problems=problems, proposal="a conflict")
        return _Decision(
            outcome=CellOutcome(
                status="conflicting",
                detail=reply.reason or "the documents give different answers",
                evidence=verified,
            )
        )

    # answered
    answer = reply.answer or ""
    if not verified and column.evidence_required:
        problems.append("no quote that appears in the chunks supports the answer")
    normalised = None
    try:
        normalised = normalise(
            column.type,
            answer,
            reason=reply.reason,
            allowed_values=column.allowed_values,
            units_hint=None
            if is_per_unit(column.name, column.prompt, *[v.quote for v in verified])
            else _unit_hint(verified, by_id),
        )
    except BadValue as exc:
        problems.append(str(exc))
    if normalised is not None and verified:
        try:
            unsupported = supported_by_quote(
                column.type, answer, [v.quote for v in verified]
            )
        except BadValue as exc:
            unsupported = str(exc)
        if unsupported:
            problems.append(unsupported)

    # On the last attempt a stray bad quote is dropped if the rest stands on its own.
    if problems and final_attempt and normalised is not None:
        only_extra_quotes = verified and len(problems) == len(evidence_problems)
        if only_extra_quotes:
            problems = []
    if problems or normalised is None:
        return _Decision(
            problems=problems or ["the answer could not be read"], proposal=answer
        )

    unscaled = _unscaled_table_amount(
        column, normalised.value, verified, by_id, reply, answer
    )
    if unscaled:
        return _Decision(
            outcome=CellOutcome(
                status="low_confidence",
                detail=unscaled,
                value=normalised.value,
                display=normalised.display,
                value_type=column.type,
                evidence=verified,
            )
        )
    weak = [v for v in verified if v.chunk_confidence <= low_confidence_at]
    if weak:
        flags = sorted({f for v in weak for f in v.chunk_flags}) or [
            "low parser confidence"
        ]
        return _Decision(
            outcome=CellOutcome(
                status="low_confidence",
                detail="the quoted passage was flagged as unreliable when the document was read: "
                + ", ".join(flags),
                value=normalised.value,
                display=normalised.display,
                value_type=column.type,
                evidence=verified,
            )
        )
    return _Decision(
        outcome=CellOutcome(
            status="answered",
            value=normalised.value,
            display=normalised.display,
            value_type=column.type,
            evidence=verified,
        )
    )


def _unscaled_table_amount(
    column: ColumnRecord,
    value: object,
    verified: list[VerifiedEvidence],
    by_id: dict[UUID, ChunkRecord],
    reply: ModelReply,
    answer: str,
) -> str | None:
    """A figure read from a table whose scale is unclear is not a confident answer.

    Money: "$78,328" from a table with no stated units could be dollars or millions, so the cell
    says it cannot be trusted. A plain number: if the table DOES state units ("in millions", or
    share counts "in thousands"), the figure may be scaled by them and we cannot tell which.
    Exempt: per-share amounts (no scale), percentages, and figures the answer itself scaled
    ("$8.0 billion"). Returns the reason, or None if the figure is fine."""
    if not isinstance(value, dict):
        return None
    tables = [
        chunk
        for v in verified
        if (chunk := by_id.get(v.chunk_id) if v.chunk_id else None) is not None
        and chunk.kind == "table"
    ]
    if not tables:
        return None
    if column.type == "currency" and value.get("scale") is None:
        if is_per_unit(
            column.name, column.prompt, answer, *[v.quote for v in verified]
        ):
            return None
        return (
            "the amount comes from a table whose units (such as 'in millions') could not be found, "
            "so it is shown as printed and its scale is unknown"
        )
    if column.type == "number" and value.get("unit") is None:
        stated = next((c.units for c in tables if c.units), None)
        if stated:
            return (
                f"the number comes from a table that states its units ({stated}), so it may be "
                "scaled by them and the stored figure is shown as printed"
            )
    return None


# --- the extractor ----------------------------------------------------------------------------


class CellExtractor:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        pool: asyncpg.Pool,
        settings: Settings,
        retrieve: Retriever,
        provider: str,
        model: str,
        recorder: Recorder = no_recorder,
    ) -> None:
        self._llm = llm
        self._pool = pool
        self._settings = settings
        self._retrieve = retrieve
        self._provider = provider
        self._model = model
        self._recorder = recorder

    def _stamp(self, outcome: CellOutcome) -> CellOutcome:
        outcome.model = self._model
        outcome.prompt_version = EXTRACT_PROMPT_VERSION
        return outcome

    async def extract(
        self,
        workspace_id: UUID,
        entity_id: UUID,
        entity_name: str,
        column: ColumnRecord,
    ) -> CellOutcome:
        """Never raises for a model, retrieval or parsing problem: those become the cell's status."""
        return self._stamp(
            await self._extract(workspace_id, entity_id, entity_name, column)
        )

    async def _extract(
        self,
        workspace_id: UUID,
        entity_id: UUID,
        entity_name: str,
        column: ColumnRecord,
    ) -> CellOutcome:
        if not await repo.list_documents(
            self._pool, workspace_id, entity_id=entity_id, limit=1
        ):
            return CellOutcome(
                status="not_found", detail="this row has no documents yet"
            )

        try:
            hits = await self._retrieve(
                workspace_id,
                f"{column.name}: {column.prompt}",
                entity_id,
                self._settings.extract_candidates,
            )
            parents = await expand_to_parents(
                self._pool, hits, self._settings.retrieval_max_parents
            )
        except Exception:  # retrieval is a dependency: its failure is the cell's status
            logger.exception("retrieval failed while filling a cell")
            return CellOutcome(
                status="pending_or_error",
                detail="searching the documents failed; try again",
            )
        if not parents:
            return CellOutcome(
                status="not_found",
                detail="no passage in this row's documents matched the question",
            )

        filenames = {h.chunk.document_id: h.filename for h in hits}
        context = build_context(
            parents, filenames, self._settings.extract_context_tokens
        )
        if not context:
            return CellOutcome(
                status="not_found",
                detail="no passage in this row's documents matched the question",
            )
        return await self._ask(column, entity_name, context)

    async def _ask(
        self, column: ColumnRecord, entity_name: str, context: list[LabelledChunk]
    ) -> CellOutcome:
        problem: str | None = None
        claimed: str | None = None
        unusable = False
        for attempt in (1, 2):
            messages = build_messages(column, entity_name, context, problem)
            try:
                raw = await ask_json(
                    self._llm,
                    messages,
                    recorder=self._recorder,
                    kind="extract",
                    provider=self._provider,
                    model=self._model,
                    prompt_version=EXTRACT_PROMPT_VERSION,
                )
            except ModelCallError as exc:
                if exc.code == "bad_output" and attempt == 1:
                    problem, unusable = "the reply was not a valid JSON object.", True
                    continue
                return CellOutcome(
                    status="pending_or_error",
                    detail=f"the model call failed ({exc.code})",
                )
            try:
                reply = ModelReply.model_validate(raw)
            except ValidationError as exc:
                first = exc.errors()[0]
                problem = f"the reply was malformed ({first['msg']})."
                unusable = True
                continue
            decision = decide(
                reply,
                column,
                context,
                low_confidence_at=self._settings.extract_low_confidence,
                final_attempt=attempt == 2,
            )
            if decision.outcome is not None:
                return decision.outcome
            unusable = False
            claimed = decision.proposal
            problem = " ".join(
                f"{p}." if not p.endswith(".") else p for p in decision.problems or []
            )
        if unusable:
            return CellOutcome(
                status="pending_or_error", detail="the model's reply was unusable twice"
            )
        shown = (claimed or "an answer")[:120]
        return CellOutcome(
            status="low_confidence",
            detail=f"the model proposed {shown!r} but could not back it with a quote from the "
            f"documents: {problem}",
        )
