"""A domain pack: a ready set of columns for one use case (credit / PE diligence, a bank, a fintech).

A pack is DATA, a JSON file, not code. Adding one needs no new logic: the same planner picks from
it, the same extraction fills it, the same checks run. Every column of a pack must be a valid
column (the same rules as one made by hand), so a bad pack file fails when it is loaded, not when a
person first clicks it.

A pack holds three kinds of thing:
    columns          questions read from the documents ("What was total revenue?")
    derived columns  computed in code from other columns ("Operating margin = operating income /
                     revenue"). They name their inputs by KEY; the inputs must be ordinary columns
                     of the same pack, so there are no cycles to untangle
    checks           identities that must hold between columns ("revenue - cost of revenue = gross
                     profit"), evaluated in code and shown on the cells
"""

import re
from decimal import Decimal
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.finance.checks import CheckKind, CheckSpec
from backend.app.finance.derive import Derivation, Display, Op, check_shape
from backend.app.matrix.models import ColumnDraft, ColumnType

_KEY = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


class DerivedDef(BaseModel):
    """How a derived pack column is computed."""

    model_config = ConfigDict(populate_by_name=True)

    op: Op
    as_: Display | None = Field(default=None, alias="as")
    inputs: list[str] = Field(min_length=2, max_length=4)  # pack column keys, in order

    def derivation(self) -> Derivation:
        return Derivation(op=self.op, **({"as": self.as_} if self.as_ else {}))

    def names_in(self, pack: Pack) -> list[str]:
        """The input columns' NAMES, in order (the matrix finds columns by name)."""
        return [c.name for k in self.inputs if (c := pack.column(k)) is not None]


class PackColumn(BaseModel):
    key: str  # stable id inside the pack: the planner and tests refer to it, names may be reworded
    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=2000)
    type: ColumnType = "text"
    allowed_values: list[str] = Field(default_factory=list)
    applicability: str | None = None
    evidence_required: bool = True
    group: str = Field(default="General", max_length=60)  # how the picker sorts it
    derived: DerivedDef | None = None  # set for a column computed from other columns

    @field_validator("key")
    @classmethod
    def _key_shape(cls, value: str) -> str:
        if not _KEY.match(value):
            raise ValueError(
                "a column key is lower-case letters, digits and underscores"
            )
        return value

    @model_validator(mode="after")
    def _is_a_valid_column(self) -> Self:
        # raises with the same message a hand-made column would get
        self.draft([uuid4() for _ in self.derived.inputs] if self.derived else None)
        return self

    def draft(self, depends_on: list[UUID] | None = None) -> ColumnDraft:
        """This column as the thing the matrix creates. A derived column needs the ids of the
        columns it reads, in the order of `derived.inputs`."""
        return ColumnDraft(
            name=self.name,
            prompt=self.prompt,
            type=self.type,
            allowed_values=self.allowed_values,
            applicability=self.applicability,
            evidence_required=self.evidence_required,
            depends_on=depends_on or [],
            derivation=self.derived.derivation() if self.derived else None,
        )


class PackCheck(BaseModel):
    """A tie-out between columns, named by key."""

    id: str
    name: str
    kind: CheckKind
    a: str
    b: str
    result: str
    tolerance: Decimal
    note: str | None = None


class Pack(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)
    columns: list[PackColumn] = Field(min_length=1, max_length=60)
    checks: list[PackCheck] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _KEY.match(value):
            raise ValueError("a pack id is lower-case letters, digits and underscores")
        return value

    @model_validator(mode="after")
    def _no_repeats(self) -> Self:
        keys = [c.key for c in self.columns]
        if len(set(keys)) != len(keys):
            raise ValueError("two columns share a key")
        names = [c.name.casefold() for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("two columns share a name")
        return self

    @model_validator(mode="after")
    def _derived_columns_make_sense(self) -> Self:
        by_key = {c.key: c for c in self.columns}
        for column in self.columns:
            if column.derived is None:
                continue
            inputs = []
            for key in column.derived.inputs:
                source = by_key.get(key)
                if source is None:
                    raise ValueError(
                        f"'{column.key}' reads '{key}', which is not in the pack"
                    )
                if source.derived is not None:
                    raise ValueError(
                        f"'{column.key}' reads '{key}', which is itself computed: a computed "
                        "column may only read ordinary columns"
                    )
                inputs.append(source)
            check_shape(column.derived.derivation(), len(inputs))
            expected = _derived_type(column.derived.op, [i.type for i in inputs])
            if column.type != expected:
                raise ValueError(
                    f"'{column.key}' should be of type '{expected}' (it is '{column.type}')"
                )
        return self

    @model_validator(mode="after")
    def _checks_name_real_columns(self) -> Self:
        keys = {c.key for c in self.columns}
        ids = [c.id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("two checks share an id")
        for check in self.checks:
            for key in (check.a, check.b, check.result):
                if key not in keys:
                    raise ValueError(
                        f"check '{check.id}' names '{key}', which is not in the pack"
                    )
        return self

    def column(self, key: str) -> PackColumn | None:
        return next((c for c in self.columns if c.key == key), None)

    def with_inputs(self, keys: list[str]) -> list[PackColumn]:
        """The chosen columns plus whatever the computed ones among them read, in pack order, so the
        inputs always come first."""
        wanted = set(keys)
        for key in keys:
            column = self.column(key)
            if column is not None and column.derived is not None:
                wanted.update(column.derived.inputs)
        return [c for c in self.columns if c.key in wanted]

    def check_specs(self) -> list[CheckSpec]:
        """The checks with column keys turned into the column names the matrix uses."""
        names = {c.key: c.name for c in self.columns}
        return [
            CheckSpec(
                id=c.id,
                name=c.name,
                kind=c.kind,
                a=names[c.a],
                b=names[c.b],
                result=names[c.result],
                tolerance=c.tolerance,
                note=c.note,
            )
            for c in self.checks
        ]


def _derived_type(op: str, input_types: list[str]) -> str:
    """The type a computed column must have: ratios, growth and net-debt multiples are plain
    numbers; a difference or sum of amounts is an amount."""
    if op in ("difference", "sum") and set(input_types) == {"currency"}:
        return "currency"
    return "number"
