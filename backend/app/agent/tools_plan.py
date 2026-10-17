"""The tools that turn a question into columns: propose_columns, list_packs, add_pack.

propose_columns is the heart of the product. Flow:

    the model calls propose_columns(question)
      -> the GATE asks the planner for columns, validates them in code, estimates the cost, and
         STORES the exact plan as a pending action. Nothing is created yet.
      -> the person sees the plan card (columns, cells, cost) and confirms or cancels
      -> on a yes, the stored plan runs: the columns are created and the fill is started

What runs is what the card showed: the stored columns, not anything the model types later. The
tool also refuses to create anything unless a person confirmed (ctx.confirmed), so a model that
passes the same arguments itself cannot skip the card.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.agent.tools import Gate, ToolContext, ToolResult, fail
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db.records import EntityRecord
from backend.app.matrix.models import ColumnRecord
from backend.app.matrix.runs import (
    InvalidCap,
    RunInProgressError,
    RunScope,
    ScopeError,
    estimate_cost,
)
from backend.app.packs.loader import get_packs
from backend.app.packs.models import Pack
from backend.app.planning.plan import ColumnPlan, PlannedColumn

DEFAULT_PACK = "credit_pe_diligence"


class ProposeArgs(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The user's question, in their own words",
    )
    pack: str | None = Field(
        default=None,
        description="A pack id to draw columns from. Leave out for the default pack.",
    )
    # Filled in by the gate and read back when a person confirms. The model is never told these
    # exist, and the tool ignores them unless a person confirmed.
    columns: list[PlannedColumn] | None = Field(
        default=None, json_schema_extra={"hidden": True}
    )
    plan: dict[str, Any] | None = Field(
        default=None, json_schema_extra={"hidden": True}
    )


def _money(value: Any) -> str:
    return "an unknown cost" if value is None else f"about ${value:.4f}"


async def make_plan(
    ctx: ToolContext, args: ProposeArgs
) -> tuple[ColumnPlan, list[EntityRecord], Pack | None] | str:
    """Ask the planner. Returns the validated plan, or a sentence saying why it cannot be made."""
    services = ctx.services
    if services.planner is None or services.matrix is None or services.runs is None:
        return "Planning columns is unavailable: no model is configured."
    packs = get_packs()
    pack = packs.get(args.pack or DEFAULT_PACK)
    if args.pack and pack is None:
        return f"There is no pack '{args.pack}'. The packs are: {', '.join(packs) or 'none'}."
    rows = await repo.list_entities(ctx.pool, ctx.workspace_id)
    existing = await mrepo.list_columns(ctx.pool, ctx.workspace_id)
    plan = await services.planner.plan(
        args.question,
        existing=existing,
        pack=pack,
        rows=[r.name for r in rows],
    )
    return plan, rows, pack


async def _cells_to_fill(
    ctx: ToolContext, plan: ColumnPlan, rows: list[EntityRecord]
) -> int:
    """Cells the MODEL will have to fill. New columns are empty for every row; a reused column only
    needs its missing cells. Computed columns are arithmetic and cost nothing, so they are not
    counted."""
    assert ctx.services.runs is not None
    cells = len(rows) * sum(
        1 for c in plan.columns if c.source != "existing" and c.derived is None
    )
    reused = [c.column_id for c in plan.columns if c.column_id is not None]
    if reused and rows:
        try:
            counted = await ctx.services.runs.plan(
                ctx.workspace_id, RunScope(column_ids=reused)
            )
            cells += (
                counted.estimate.cells
            )  # the run's own count already leaves computed cells out
        except ScopeError, InvalidCap:
            pass  # the run itself will report it
    return cells


async def propose_gate(ctx: ToolContext, args: ProposeArgs) -> Gate | None:
    made = await make_plan(ctx, args)
    if isinstance(made, str):
        return None  # the tool reports why
    plan, rows, _ = made
    if not plan.columns:
        return None  # nothing to confirm: the tool says what went wrong

    cells = await _cells_to_fill(ctx, plan, rows)
    estimate = await estimate_cost(ctx.pool, ctx.settings, cells)
    new = [c for c in plan.columns if c.source != "existing"]
    computed_count = sum(1 for c in new if c.derived is not None)
    cap = ctx.settings.run_default_cap_usd
    payload: dict[str, Any] = {
        "question": args.question,
        "columns": [c.model_dump(mode="json") for c in plan.columns],
        "rows": len(rows),
        "cells": cells,
        "estimate_usd": None
        if estimate.expected_usd is None
        else str(estimate.expected_usd),
        "basis": estimate.basis,
        "cap_usd": str(cap),
        "notes": plan.notes,
    }
    names = ", ".join(c.name for c in plan.columns)
    what = (
        f"Add {len(new)} column(s) and fill {cells} cells for {len(rows)} companies"
        if rows
        else f"Add {len(new)} column(s). There are no companies yet, so nothing will be filled"
    )
    reused = len(plan.columns) - len(new)
    summary = (
        f"{what}: {names}. "
        + (
            f"{computed_count} of them are computed in code, at no cost. "
            if computed_count
            else ""
        )
        + (f"{reused} column(s) already exist and are reused. " if reused else "")
        + f"Cost {_money(estimate.expected_usd)}; it stops by itself at ${cap}."
    )
    stored = {
        "question": args.question,
        "pack": args.pack,
        "columns": payload["columns"],
        "plan": payload,
    }
    return Gate(risk="costly", summary=summary, plan=payload, args=stored)


async def propose_columns(ctx: ToolContext, args: ProposeArgs) -> ToolResult:
    if not args.columns:
        # Not confirmed, or nothing could be planned. Say which.
        made = await make_plan(ctx, args)
        if isinstance(made, str):
            return fail(made)
        notes = "; ".join(made[0].notes)
        return fail(
            "I could not turn that into columns"
            + (f" ({notes})" if notes else "")
            + ". Ask for specific facts, for example 'revenue and free cash flow'."
        )
    if not ctx.confirmed:
        return fail(
            "This plan has to be confirmed by the person first. Nothing was created."
        )

    matrix, runs = ctx.services.matrix, ctx.services.runs
    if matrix is None or runs is None:
        return fail(
            "Creating columns is unavailable: no extraction model is configured."
        )

    ids: list[UUID] = []
    created: list[str] = []
    known = {
        c.name.casefold(): c.id
        for c in await mrepo.list_columns(ctx.pool, ctx.workspace_id)
    }
    for column in args.columns:
        if column.source == "existing" and column.column_id is not None:
            ids.append(column.column_id)
            continue
        depends_on: list[UUID] | None = None
        if column.derived is not None:
            try:
                depends_on = [known[name.casefold()] for name in column.inputs]
            except KeyError as missing:
                return fail(
                    f"Could not create '{column.name}': the column {missing.args[0]!r} it is "
                    "computed from does not exist."
                )
        try:
            record: ColumnRecord = await matrix.create_column(
                ctx.workspace_id, column.draft(depends_on)
            )
        except mrepo.DuplicateColumnError:
            # It appeared since the plan was shown (made by hand, say). Use it.
            found = next(
                (
                    c
                    for c in await mrepo.list_columns(ctx.pool, ctx.workspace_id)
                    if c.name.casefold() == column.name.casefold()
                ),
                None,
            )
            if found is None:
                return fail(f"Could not create the column '{column.name}'.")
            ids.append(found.id)
            known[found.name.casefold()] = found.id
            continue
        ids.append(record.id)
        known[record.name.casefold()] = record.id
        created.append(record.name)

    data: dict[str, Any] = {
        "question": args.question,
        "columns": [c.name for c in args.columns],
    }
    head = f"Added {len(created)} column(s)" + (
        f" and reused {len(ids) - len(created)}" if len(ids) > len(created) else ""
    )
    if not await repo.list_entities(ctx.pool, ctx.workspace_id):
        return ToolResult(
            ok=True,
            text=f"{head}. There are no companies yet, so nothing was filled.",
            summary=head,
            changed_matrix=True,
            data=data,
        )
    try:
        run = await runs.start(
            ctx.workspace_id, RunScope(column_ids=ids), confirmed=True
        )
    except RunInProgressError:
        return ToolResult(
            ok=True,
            text=f"{head}. A fill is already going in this workspace, so these were not filled yet: "
            "use 'Fill empty cells' when it ends.",
            summary=head,
            changed_matrix=True,
            data=data,
        )
    except (ScopeError, InvalidCap) as exc:
        return fail(f"{head}, but the fill could not start: {exc}")
    return ToolResult(
        ok=True,
        text=f"{head}, and started filling {run.total} cells.",
        summary=f"{head}; filling {run.total} cells",
        changed_matrix=True,
        data={**data, "run_id": str(run.id)},
    )


# --- packs ------------------------------------------------------------------------------------------


class NoArgs(BaseModel):
    pass


async def list_packs(ctx: ToolContext, _: NoArgs) -> ToolResult:
    lines = []
    for pack in get_packs().values():
        groups = sorted({c.group for c in pack.columns})
        lines.append(
            f"- {pack.id}: {pack.name}. {pack.description} "
            f"({len(pack.columns)} columns: {', '.join(groups)})"
        )
    return ToolResult(
        ok=True, text="\n".join(lines) or "No packs.", summary=f"{len(lines)} pack(s)"
    )


class AddPackArgs(BaseModel):
    pack: str = Field(description="The pack id")
    keys: list[str] | None = Field(
        default=None,
        description="Only these column keys of the pack. Leave out for the whole pack.",
    )


async def add_pack(ctx: ToolContext, args: AddPackArgs) -> ToolResult:
    if ctx.services.matrix is None:
        return fail("Columns are unavailable: no extraction model is configured.")
    packs = get_packs()
    pack = packs.get(args.pack)
    if pack is None:
        return fail(
            f"There is no pack '{args.pack}'. The packs are: {', '.join(packs)}."
        )
    wanted = pack.columns
    if args.keys is not None:
        unknown = [k for k in args.keys if pack.column(k) is None]
        if unknown:
            return fail(f"The pack has no column(s) {', '.join(unknown)}.")
        wanted = pack.with_inputs(
            args.keys
        )  # what a computed column reads comes with it
    known = {
        c.name.casefold(): c.id
        for c in await mrepo.list_columns(ctx.pool, ctx.workspace_id)
    }
    added: list[str] = []
    for column in wanted:  # pack order: ordinary columns come before the computed ones
        if column.name.casefold() in known:
            continue
        depends_on: list[UUID] | None = None
        if column.derived is not None:
            names = column.derived.names_in(pack)
            if any(n.casefold() not in known for n in names):
                continue  # an input could not be added: leave this one for later
            depends_on = [known[n.casefold()] for n in names]
        try:
            record = await ctx.services.matrix.create_column(
                ctx.workspace_id, column.draft(depends_on)
            )
        except mrepo.DuplicateColumnError:
            continue
        known[record.name.casefold()] = record.id
        added.append(column.name)
    skipped = len(wanted) - len(added)
    text = f"Added {len(added)} column(s) from '{pack.name}'" + (
        f"; {skipped} already existed." if skipped else "."
    )
    return ToolResult(
        ok=True,
        text=text + " They are empty until filled.",
        summary=text,
        changed_matrix=bool(added),
    )
