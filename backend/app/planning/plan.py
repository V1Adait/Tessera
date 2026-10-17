"""Turning the model's proposed columns into a plan the code is willing to stand behind.

The model PROPOSES columns for a question. This module decides what is accepted, and it is pure
(no model, no database), so every rule can be tested:

    a column of the pack (by key)   -> the pack's own wording is used, not the model's rewording
    a computed pack column          -> the columns it reads are pulled in first ("operating margin"
                                       brings "operating income" and "revenue")
    a column the model invented     -> must pass the same rules as a hand-made column
    a column that already exists    -> reused (its empty cells get filled), never created twice
    repeats, too many, junk         -> dropped, and the reason is recorded in `notes`

Nothing is created here. The person sees the plan first.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from backend.app.finance.derive import Derivation
from backend.app.matrix.models import ColumnDraft, ColumnRecord, ColumnType
from backend.app.packs.models import Pack, PackColumn

MAX_COLUMNS = 8


class PlannedColumn(BaseModel):
    name: str
    prompt: str
    type: ColumnType = "text"
    allowed_values: list[str] = Field(default_factory=list)
    applicability: str | None = None
    evidence_required: bool = True
    source: Literal["pack", "new", "existing"]
    pack_key: str | None = None
    column_id: UUID | None = None  # set when source is "existing"
    # A column computed in code: how, and the NAMES of the columns it reads, in order.
    derived: Derivation | None = None
    inputs: list[str] = Field(default_factory=list)

    def draft(self, depends_on: list[UUID] | None = None) -> ColumnDraft:
        """The column as the matrix creates it. A computed column needs its inputs' ids."""
        return ColumnDraft(
            name=self.name,
            prompt=self.prompt,
            type=self.type,
            allowed_values=self.allowed_values,
            applicability=self.applicability,
            evidence_required=self.evidence_required,
            depends_on=depends_on or [],
            derivation=self.derived,
        )


class ColumnPlan(BaseModel):
    question: str
    columns: list[PlannedColumn] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list
    )  # what was dropped or changed, and why


def _first_error(exc: ValidationError) -> str:
    return str(exc.errors()[0]["msg"]).removeprefix("Value error, ")


def _as_str_list(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def resolve_plan(
    raw: Any,
    *,
    question: str,
    pack: Pack | None,
    existing: list[ColumnRecord],
    max_columns: int = MAX_COLUMNS,
) -> ColumnPlan:
    """The model's reply (any shape) -> an honest plan. Never raises."""
    plan = ColumnPlan(question=question)
    proposed = raw.get("columns") if isinstance(raw, dict) else None
    if not isinstance(proposed, list):
        plan.notes.append("the planner's reply had no list of columns")
        return plan

    chosen: list[PlannedColumn] = []
    for item in proposed:
        if not isinstance(item, dict):
            plan.notes.append("ignored a proposed column that was not an object")
            continue
        planned = _planned_from(item, pack, plan.notes)
        if planned is not None:
            chosen.append(planned)

    by_name = {c.name.casefold(): c for c in existing}
    ordered = _with_inputs(chosen, pack, plan.notes)

    seen: set[str] = set()
    for planned in ordered:
        key = planned.name.casefold()
        if key in seen:
            plan.notes.append(f"'{planned.name}' was proposed twice; kept once")
            continue
        seen.add(key)
        match = by_name.get(key)
        if match is not None:
            planned = planned.model_copy(
                update={"source": "existing", "column_id": match.id}
            )
        plan.columns.append(planned)

    if len(plan.columns) > max_columns:
        cut = plan.columns[max_columns:]
        plan.notes.append(
            f"{len(plan.columns)} columns were proposed; kept the first {max_columns} and left out "
            + ", ".join(f"'{c.name}'" for c in cut)
        )
        plan.columns = plan.columns[:max_columns]
    _drop_computed_without_inputs(plan, by_name)
    return plan


def _from_pack(column: PackColumn, pack: Pack) -> PlannedColumn:
    d = column.draft(
        [UUID(int=i + 1) for i in range(len(column.derived.inputs))]
        if column.derived
        else None
    )
    return PlannedColumn(
        name=d.name,
        prompt=d.prompt,
        type=d.type,
        allowed_values=d.allowed_values,
        applicability=d.applicability,
        evidence_required=d.evidence_required,
        source="pack",
        pack_key=column.key,
        derived=column.derived.derivation() if column.derived else None,
        inputs=column.derived.names_in(pack) if column.derived else [],
    )


def _planned_from(
    item: dict[str, Any], pack: Pack | None, notes: list[str]
) -> PlannedColumn | None:
    pack_key = item.get("pack_key")
    if isinstance(pack_key, str) and pack_key:
        from_pack = pack.column(pack_key) if pack is not None else None
        if from_pack is not None and pack is not None:
            return _from_pack(from_pack, pack)
        notes.append(
            f"'{pack_key}' is not a column of the pack; treated as a new column"
        )

    try:
        draft = ColumnDraft(
            name=item.get("name"),  # type: ignore[arg-type]  # validated just below
            prompt=item.get("prompt"),  # type: ignore[arg-type]
            type=item.get("type", "text"),
            allowed_values=_as_str_list(item.get("allowed_values")),
        )
    except ValidationError as exc:
        label = item.get("name") if isinstance(item.get("name"), str) else "a column"
        notes.append(f"dropped '{label}': {_first_error(exc)}")
        return None
    return PlannedColumn(
        name=draft.name,
        prompt=draft.prompt,
        type=draft.type,
        allowed_values=draft.allowed_values,
        applicability=draft.applicability,
        evidence_required=draft.evidence_required,
        source="new",
    )


def _with_inputs(
    chosen: list[PlannedColumn], pack: Pack | None, notes: list[str]
) -> list[PlannedColumn]:
    """Each computed column comes after the columns it reads. A column it reads that was not chosen
    is added (and the note says so), because the computed column is empty without it."""
    if pack is None:
        return chosen
    result: list[PlannedColumn] = []
    present = {c.name.casefold() for c in chosen}
    queued: set[str] = set()
    for planned in chosen:
        if planned.derived is not None and planned.pack_key is not None:
            column = pack.column(planned.pack_key)
            for key in column.derived.inputs if column and column.derived else []:
                source = pack.column(key)
                if source is None or source.name.casefold() in queued:
                    continue
                if source.name.casefold() not in present:
                    notes.append(
                        f"added '{source.name}' because '{planned.name}' is computed from it"
                    )
                    present.add(source.name.casefold())
                if source.name.casefold() not in {c.name.casefold() for c in result}:
                    queued.add(source.name.casefold())
                    result.append(_from_pack(source, pack))
        # A repeat of something the person chose stays, so the repeat is reported; only a column
        # this function itself added as an input is skipped here.
        if planned.name.casefold() not in queued:
            result.append(planned)
    return result


def _drop_computed_without_inputs(
    plan: ColumnPlan, by_name: dict[str, ColumnRecord]
) -> None:
    """After cutting to the column limit, a computed column whose inputs did not all survive (and do
    not already exist) would be empty: drop it and say so."""
    kept = {c.name.casefold() for c in plan.columns} | set(by_name)
    survivors: list[PlannedColumn] = []
    for column in plan.columns:
        missing = [n for n in column.inputs if n.casefold() not in kept]
        if column.source != "existing" and column.derived is not None and missing:
            plan.notes.append(
                f"dropped '{column.name}': it needs {', '.join(repr(m) for m in missing)}, which did not fit"
            )
            continue
        survivors.append(column)
    plan.columns = survivors
