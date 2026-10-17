"""Cells that are computed in code, and the tie-outs on a row. No model, no cost.

    compute_cell   a derived column's cell for one row: read the input cells, check they can be
                   combined (finance/derive.py), save the result with the inputs' own quotes as its
                   evidence, so even a computed number can be traced to words in a document
    refresh_row    after any cell of a row changes: recompute that row's derived cells, then re-run
                   the tie-out checks and store their results on the cells they are about

Both are idempotent: running them twice gives the same cells. They are cheap, so they run after
every fill instead of tracking exactly what changed.
"""

import logging
from typing import Any
from uuid import UUID

import asyncpg

from backend.app.db import matrix_repository as mrepo
from backend.app.finance.checks import CheckSpec, run_checks
from backend.app.finance.derive import Derived, Operand, compute
from backend.app.finance.periods import parse_period
from backend.app.matrix.models import (
    MAX_EVIDENCE,
    CellOutcome,
    CellRecord,
    ColumnRecord,
    VerifiedEvidence,
)

logger = logging.getLogger(__name__)

COMPUTED_VERSION = (
    "derived-1"  # stored as the cell's prompt_version: there is no prompt
)


def operand_of(column: ColumnRecord, cell: CellRecord | None) -> Operand:
    """One input cell, as the calculation and the checks see it."""
    if cell is None:
        return Operand(column.name, "not_filled", None, column.type, None)
    return Operand(
        column.name,
        cell.status,
        cell.value,
        column.type,
        parse_period(cell.period),
    )


def _evidence_of(inputs: list[CellRecord | None]) -> list[VerifiedEvidence]:
    """The quotes behind a computed number: each input's first quote, then any further ones."""
    firsts: list[VerifiedEvidence] = []
    rest: list[VerifiedEvidence] = []
    for cell in inputs:
        if cell is None:
            continue
        for index, e in enumerate(cell.evidence):
            item = VerifiedEvidence(
                document_id=e.document_id,
                chunk_id=e.chunk_id,
                quote=e.quote,
                char_start=e.char_start,
                char_end=e.char_end,
                exact_range=e.exact_range,
                page_start=e.page_start,
                page_end=e.page_end,
            )
            (firsts if index == 0 else rest).append(item)
    return (firsts + rest)[:MAX_EVIDENCE]


def outcome_of(
    column: ColumnRecord,
    derived: Derived,
    operands: list[Operand],
    inputs: list[CellRecord | None],
) -> CellOutcome:
    """The result of a calculation as a cell outcome."""
    assert column.derivation is not None
    trace: dict[str, Any] = {
        "op": column.derivation.op,
        "inputs": [
            {
                "column": o.name,
                "display": c.display_value if c is not None else None,
                "period": o.period.label if o.period else None,
            }
            for o, c in zip(operands, inputs, strict=True)
        ],
        "notes": derived.notes,
    }
    evidence = _evidence_of(inputs) if derived.status == "answered" else []
    if derived.status == "answered" and column.evidence_required and not evidence:
        # Every answered cell has evidence. A computed number whose inputs have none has none either.
        return CellOutcome(
            status="low_confidence",
            detail="its inputs have no quotes behind them, so the result cannot be traced",
            prompt_version=COMPUTED_VERSION,
            derivation=trace,
        )
    return CellOutcome(
        status=derived.status,
        detail=derived.detail,
        value=derived.value,
        display=derived.display,
        value_type=column.type,
        evidence=evidence,
        prompt_version=COMPUTED_VERSION,
        period=derived.period.label if derived.period else None,
        derivation=trace,
    )


async def compute_cell(
    pool: asyncpg.Pool,
    workspace_id: UUID,
    entity_id: UUID,
    column: ColumnRecord,
    columns: list[ColumnRecord],
    *,
    replace_pinned: bool = False,
) -> CellRecord:
    """Compute and save one derived cell. Raises mrepo.CellPinnedError for a person's cell."""
    assert column.derivation is not None
    by_id = {c.id: c for c in columns}
    sources = [by_id.get(cid) for cid in column.depends_on]
    if any(s is None for s in sources):
        outcome = CellOutcome(
            status="pending_or_error",
            detail="a column this one is computed from was removed",
            prompt_version=COMPUTED_VERSION,
        )
    else:
        cells = {
            c.column_id: c
            for c in await mrepo.list_cells(pool, workspace_id, entity_id=entity_id)
        }
        known = [s for s in sources if s is not None]
        inputs = [cells.get(s.id) for s in known]
        operands = [operand_of(s, c) for s, c in zip(known, inputs, strict=True)]
        outcome = outcome_of(
            column, compute(column.derivation, operands), operands, inputs
        )
    return await mrepo.save_cell(
        pool,
        workspace_id=workspace_id,
        entity_id=entity_id,
        column=column,
        outcome=outcome,
        replace_pinned=replace_pinned,
    )


async def refresh_row(
    pool: asyncpg.Pool,
    workspace_id: UUID,
    entity_id: UUID,
    check_specs: list[CheckSpec],
) -> None:
    """Bring a row's computed cells and tie-outs up to date with its other cells."""
    columns = await mrepo.list_columns(pool, workspace_id)
    derived = [c for c in columns if c.derivation is not None]
    for _ in range(
        2 if derived else 0
    ):  # a computed column may read another computed column
        for column in derived:
            try:
                await compute_cell(pool, workspace_id, entity_id, column, columns)
            except mrepo.CellPinnedError:
                continue  # a person set this cell: leave it
    await _refresh_checks(pool, workspace_id, entity_id, columns, check_specs)


async def _refresh_checks(
    pool: asyncpg.Pool,
    workspace_id: UUID,
    entity_id: UUID,
    columns: list[ColumnRecord],
    check_specs: list[CheckSpec],
) -> None:
    cells = {
        c.column_id: c
        for c in await mrepo.list_cells(pool, workspace_id, entity_id=entity_id)
    }
    operands = {c.name.casefold(): operand_of(c, cells.get(c.id)) for c in columns}
    shown: dict[str, list[dict[str, str]]] = {}
    for result in run_checks(check_specs, operands):
        shown.setdefault(result.attach_to.casefold(), []).append(result.stored())
    for column in columns:
        cell = cells.get(column.id)
        if cell is None:
            continue
        wanted = shown.get(column.name.casefold(), [])
        if wanted != cell.checks:
            await mrepo.set_checks(pool, cell.id, wanted)
