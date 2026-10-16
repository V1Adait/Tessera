"""Filling many cells: plan it, price it, ask, then run it safely in the background.

    scope (some rows, some columns)  ->  plan: which cells actually need work, in reading order
        ->  estimate: about N cells, roughly $X, from what past calls really cost
        ->  confirm (when it is big or dear)  ->  run: a few cells at a time, until done, stopped,
            or the spending cap is reached  ->  the cells, and the run's own record (progress, spend)

A run is NOT "refill the whole table". Its scope names the rows and columns it may touch, and by
default it only does cells that are missing, stale (their column changed) or ended in an error. A
cell that is already answered is left alone, and a cell a person pinned is never touched. A question
about one company is a run with one row in scope.

What a run guarantees:
  * nothing is spent before the estimate has been shown, when the run is big or dear
  * it stops by itself at its cap; because a few cells are in flight at once it can overshoot by at
    most the cells running at that moment, and it says what it actually spent
  * one failing cell does not stop the run: it becomes a pending_or_error cell with a reason
  * a restart does not lose finished work, and a run cut off by one is marked `interrupted`
"""

import asyncio
import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Literal
from uuid import UUID

import asyncpg
from pydantic import BaseModel, Field, field_validator

from backend.app.config import Settings
from backend.app.costs import get_price_book
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db import runs_repository as rrepo
from backend.app.db.runs_repository import RunInProgressError, RunRecord
from backend.app.matrix.models import CellStatus
from backend.app.matrix.service import MatrixService

logger = logging.getLogger(__name__)

TYPICAL_PROMPT_TOKENS = (
    7000  # what one extraction call reads, measured on the Apple 10-K
)
TYPICAL_REPLY_TOKENS = 200
MIN_SAMPLES = 5  # fewer measured calls than this is too little to average
MAX_IDS = 1000

__all__ = ["RunInProgressError", "RunRecord"]


class ScopeError(LookupError):
    """The scope names a row or a column that is not in this workspace."""


class ConfirmationRequired(Exception):
    """The run is big or costly enough to need a yes first. `plan` is what to show the person."""

    def __init__(self, plan: RunPlan) -> None:
        super().__init__(plan.confirmation_reason or "confirmation required")
        self.plan = plan


class InvalidCap(ValueError):
    """The spending cap asked for is not allowed."""


_DEFAULT_REDO: list[CellStatus] = ["pending_or_error"]


# --- scope and plan ---------------------------------------------------------------------------


class RunScope(BaseModel):
    """Which cells a run may touch. Leave a list out to mean "all"; give it to narrow."""

    entity_ids: list[UUID] | None = Field(default=None, max_length=MAX_IDS)  # rows
    column_ids: list[UUID] | None = Field(default=None, max_length=MAX_IDS)
    missing: bool = True  # cells that do not exist yet
    stale: bool = True  # cells whose column changed since they were filled
    statuses: list[CellStatus] = Field(default_factory=lambda: _DEFAULT_REDO.copy())
    include_pinned: bool = False  # cells a person set: left alone unless this is true
    refresh: bool = (
        False  # ignore stored model replies and ask again (and store the new ones)
    )

    @field_validator("entity_ids", "column_ids")
    @classmethod
    def _not_empty(cls, ids: list[UUID] | None) -> list[UUID] | None:
        if ids is not None and not ids:
            raise ValueError("give at least one id, or leave the list out to mean all")
        return ids


class PlannedCell(BaseModel):
    entity_id: UUID
    column_id: UUID
    reason: str  # missing | stale | the status being redone


class CostEstimate(BaseModel):
    cells: int
    per_cell_usd: Decimal | None
    expected_usd: Decimal | None
    basis: Literal["measured", "price_list", "unknown"]
    samples: int  # how many past calls the average comes from


class RunPlan(BaseModel):
    workspace_id: UUID
    scope: RunScope
    cells: list[PlannedCell]
    skipped_pinned: int  # in scope, would qualify, but a person set them
    estimate: CostEstimate
    cap_usd: Decimal
    needs_confirmation: bool
    confirmation_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            counts[cell.reason] = counts.get(cell.reason, 0) + 1
        return counts


def choose_cells(
    entities: list[UUID],
    columns: list[UUID],
    states: dict[tuple[UUID, UUID], tuple[str, bool, bool]],
    scope: RunScope,
) -> tuple[list[PlannedCell], int]:
    """The cells in scope that need work, rows first then columns, and how many were skipped
    because a person pinned them. Pure: the decisions are made here, from the cell states."""
    redo = set(scope.statuses)
    chosen: list[PlannedCell] = []
    pinned = 0
    for entity_id in entities:
        for column_id in columns:
            state = states.get((entity_id, column_id))
            if state is None:
                reason = "missing" if scope.missing else None
            else:
                status, stale, is_pinned = state
                if stale and scope.stale:
                    reason = "stale"
                elif status in redo:
                    reason = status
                else:
                    reason = None
                if reason is not None and is_pinned and not scope.include_pinned:
                    pinned += 1
                    continue
            if reason is not None:
                chosen.append(
                    PlannedCell(entity_id=entity_id, column_id=column_id, reason=reason)
                )
    return chosen, pinned


async def estimate_cost(db: rrepo.Db, settings: Settings, cells: int) -> CostEstimate:
    """About what `cells` cells will cost. From what this model's recent calls really cost when
    there are enough of them, else from its published price, else honestly unknown."""
    factor = Decimal(str(settings.run_calls_per_cell))
    average, samples = await rrepo.measured_cost_per_call(db, settings.extract_model)
    if average is not None and samples >= MIN_SAMPLES:
        per_cell, basis = average * factor, "measured"
    else:
        price = get_price_book().get(settings.extract_provider, settings.extract_model)
        if price is not None:
            per_call = (
                price.prompt * TYPICAL_PROMPT_TOKENS
                + price.completion * TYPICAL_REPLY_TOKENS
            )
            per_cell, basis = per_call * factor, "price_list"
        else:
            per_cell, basis = None, "unknown"
    return CostEstimate(
        cells=cells,
        per_cell_usd=per_cell,
        expected_usd=per_cell * cells if per_cell is not None else None,
        basis=basis,  # type: ignore[arg-type]
        samples=samples,
    )


# --- running -----------------------------------------------------------------------------------


class _Live:
    """The counters of one run while it is going."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.spent = Decimal(0)
        self.unknown = 0
        self.cap_hit = False
        self.lock = asyncio.Lock()
        self.stop = asyncio.Event()


class RunManager:
    def __init__(
        self, pool: asyncpg.Pool, settings: Settings, service: MatrixService
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._service = service
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._live: dict[UUID, _Live] = {}

    # --- planning ---

    def _check_cap(self, cap: Decimal | None) -> Decimal:
        chosen = self._settings.run_default_cap_usd if cap is None else cap
        if chosen <= 0:
            raise InvalidCap("the cap must be more than zero")
        if chosen > self._settings.run_max_cap_usd:
            raise InvalidCap(
                f"the cap cannot be more than ${self._settings.run_max_cap_usd}"
            )
        return chosen

    async def plan(
        self, workspace_id: UUID, scope: RunScope, cap_usd: Decimal | None = None
    ) -> RunPlan:
        """What a run with this scope would do and cost. Spends nothing. Raises ScopeError,
        InvalidCap, or WorkspaceNotFoundError."""
        cap = self._check_cap(cap_usd)
        if await repo.get_workspace(self._pool, workspace_id) is None:
            raise repo.WorkspaceNotFoundError(str(workspace_id))
        entities = [e.id for e in await repo.list_entities(self._pool, workspace_id)]
        columns = [c.id for c in await mrepo.list_columns(self._pool, workspace_id)]
        for label, wanted, known in (
            ("rows", scope.entity_ids, entities),
            ("columns", scope.column_ids, columns),
        ):
            unknown = set(wanted or []) - set(known)
            if unknown:
                raise ScopeError(
                    f"{len(unknown)} of the {label} named are not in this workspace"
                )
        wanted_rows = set(scope.entity_ids) if scope.entity_ids is not None else None
        wanted_cols = set(scope.column_ids) if scope.column_ids is not None else None
        rows = [i for i in entities if wanted_rows is None or i in wanted_rows]
        cols = [i for i in columns if wanted_cols is None or i in wanted_cols]
        states = await rrepo.cell_states(self._pool, workspace_id)
        cells, pinned = choose_cells(rows, cols, states, scope)
        estimate = await estimate_cost(self._pool, self._settings, len(cells))

        reasons: list[str] = []
        if estimate.expected_usd is None:
            reasons.append("its cost cannot be estimated")
        elif estimate.expected_usd > self._settings.run_confirm_above_usd:
            reasons.append(f"it is estimated at about ${estimate.expected_usd:.4f}")
        if len(cells) > self._settings.run_confirm_above_cells:
            reasons.append(f"it fills {len(cells)} cells")
        warnings: list[str] = []
        if estimate.expected_usd is not None and estimate.expected_usd > cap:
            warnings.append(
                f"the estimate (${estimate.expected_usd:.4f}) is above the cap (${cap}): "
                "the run will stop before it finishes"
            )
        if pinned:
            warnings.append(
                f"{pinned} cell(s) set by a person will be left as they are"
            )
        return RunPlan(
            workspace_id=workspace_id,
            scope=scope,
            cells=cells,
            skipped_pinned=pinned,
            estimate=estimate,
            cap_usd=cap,
            needs_confirmation=bool(cells) and bool(reasons),
            confirmation_reason=("; ".join(reasons) or None) if cells else None,
            warnings=warnings,
        )

    # --- starting and stopping ---

    async def start(
        self,
        workspace_id: UUID,
        scope: RunScope,
        *,
        confirmed: bool = False,
        cap_usd: Decimal | None = None,
    ) -> RunRecord:
        """Plan, and if allowed, start filling in the background. Raises ConfirmationRequired
        (with the plan) when a yes is needed first, RunInProgressError, ScopeError, InvalidCap."""
        plan = await self.plan(workspace_id, scope, cap_usd)
        if plan.needs_confirmation and not confirmed:
            raise ConfirmationRequired(plan)
        record = await rrepo.create_run(
            self._pool,
            workspace_id,
            scope.model_dump(mode="json"),
            total=len(plan.cells),
            cap_usd=plan.cap_usd,
            estimate_usd=plan.estimate.expected_usd,
            status="running" if plan.cells else "completed",
        )
        if plan.cells:
            live = _Live(len(plan.cells))
            self._live[record.id] = live
            task = asyncio.ensure_future(self._execute(record, plan, live))
            self._tasks[record.id] = task
            task.add_done_callback(self._forgetter(record.id))
        return record

    def _forgetter(self, run_id: UUID) -> Callable[[asyncio.Task[None]], None]:
        def forget(_task: asyncio.Task[None]) -> None:
            self._forget(run_id)

        return forget

    def _forget(self, run_id: UUID) -> None:
        self._tasks.pop(run_id, None)
        self._live.pop(run_id, None)

    async def stop(self, workspace_id: UUID, run_id: UUID) -> RunRecord:
        """Ask a run to stop. Cells already being filled finish; the rest are not started. Raises
        LookupError for an unknown run."""
        record = await rrepo.get_run(self._pool, run_id)
        if record is None or record.workspace_id != workspace_id:
            raise LookupError(str(run_id))
        live = self._live.get(run_id)
        if live is not None:
            live.stop.set()
        return record

    async def get(self, workspace_id: UUID, run_id: UUID) -> RunRecord | None:
        record = await rrepo.get_run(self._pool, run_id)
        return (
            record
            if record is not None and record.workspace_id == workspace_id
            else None
        )

    async def recent(self, workspace_id: UUID) -> list[RunRecord]:
        return await rrepo.list_runs(self._pool, workspace_id)

    async def recover(self) -> int:
        return await rrepo.recover_interrupted_runs(self._pool)

    async def close(self, timeout: float = 10.0) -> None:
        """At shutdown: stop starting cells, give those in flight a moment, then cancel."""
        for live in self._live.values():
            live.stop.set()
        tasks = list(self._tasks.values())
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # --- the work ---

    async def _execute(self, record: RunRecord, plan: RunPlan, live: _Live) -> None:
        semaphore = asyncio.Semaphore(max(1, self._settings.run_concurrency))
        status: str = "failed"
        detail: str | None = "the run broke before it could finish"
        try:
            await asyncio.gather(
                *[self._one(record, cell, semaphore, live) for cell in plan.cells]
            )
            if live.stop.is_set() and live.skipped:
                status, detail = (
                    "stopped",
                    f"stopped with {live.skipped} cell(s) not started",
                )
            elif live.cap_hit:
                status = "cost_cap"
                detail = f"reached the ${record.cap_usd} cap with {live.skipped} cell(s) not started"
            else:
                status, detail = "completed", None
        except asyncio.CancelledError:
            status, detail = (
                "interrupted",
                "the server stopped while this run was going",
            )
            raise
        except Exception:
            logger.exception("a matrix run broke", extra={"run_id": str(record.id)})
        finally:
            try:
                await self._save(record.id, live)
                await rrepo.finish_run(self._pool, record.id, status, detail)
            except Exception:
                logger.exception(
                    "could not close the run record", extra={"run_id": str(record.id)}
                )

    async def _save(self, run_id: UUID, live: _Live) -> None:
        await rrepo.update_progress(
            self._pool,
            run_id,
            done=live.done,
            failed=live.failed,
            skipped=live.skipped,
            spent_usd=live.spent,
            unknown_cost_calls=live.unknown,
        )

    async def _one(
        self,
        record: RunRecord,
        cell: PlannedCell,
        semaphore: asyncio.Semaphore,
        live: _Live,
    ) -> None:
        async with semaphore:
            async with live.lock:
                if live.stop.is_set():
                    live.skipped += 1
                    await self._save(record.id, live)
                    return
                if live.spent >= record.cap_usd:
                    live.cap_hit = True
                    live.skipped += 1
                    await self._save(record.id, live)
                    return
            cost, unknown, failed, skipped = Decimal(0), 0, False, False
            try:
                filled = await asyncio.wait_for(
                    self._service.fill_cell(
                        record.workspace_id,
                        cell.entity_id,
                        cell.column_id,
                        refresh=bool(record.scope.get("refresh")),
                    ),
                    self._settings.run_cell_timeout_s,
                )
                cost, unknown = filled.cost_usd, filled.unknown_cost_calls
                failed = filled.cell.status == "pending_or_error"
            except mrepo.CellPinnedError:
                skipped = True  # a person pinned it after the plan was made
            except Exception as exc:  # noqa: BLE001  # one bad cell must not stop the run
                logger.warning("a cell in a run failed: %s", type(exc).__name__)
                failed = True
                reason = (
                    "the run gave up on this cell after the time limit"
                    if isinstance(exc, TimeoutError)
                    else "filling this cell failed unexpectedly"
                )
                try:
                    await self._service.record_failure(
                        record.workspace_id, cell.entity_id, cell.column_id, reason
                    )
                except Exception:
                    logger.exception("could not record a failed cell")
            async with live.lock:
                if skipped:
                    live.skipped += 1
                else:
                    live.done += 1
                    live.failed += failed
                live.spent += cost
                live.unknown += unknown
                await self._save(record.id, live)
