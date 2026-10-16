"""HTTP routes for matrix runs: preview a run, start it, watch it, stop it.

    POST /workspaces/{id}/runs/plan       what would run, and about what it would cost. Spends nothing.
    POST /workspaces/{id}/runs            start it. A big or costly run answers 409 with the plan until
                                          the request repeats with "confirmed": true.
    GET  /workspaces/{id}/runs[/{run}]    progress, spend, and how it ended
    POST /workspaces/{id}/runs/{run}/stop cells already being filled finish; the rest are not started

Thin handlers: planning, pricing and running live in matrix/runs.py.
"""

from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.app.db import repository as repo
from backend.app.matrix.runs import (
    ConfirmationRequired,
    CostEstimate,
    InvalidCap,
    PlannedCell,
    RunInProgressError,
    RunManager,
    RunPlan,
    RunRecord,
    RunScope,
    ScopeError,
)

router = APIRouter(tags=["runs"])

SAMPLE_CELLS = (
    50  # a plan can name thousands of cells: the preview lists only the first few
)


def get_runs(request: Request) -> RunManager:
    manager: RunManager | None = getattr(request.app.state, "runs", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="runs are not configured (needs the database and an extraction model key)",
        )
    return manager


RunsDep = Annotated[RunManager, Depends(get_runs)]


class PlanRequest(BaseModel):
    scope: RunScope = Field(default_factory=RunScope)
    max_cost_usd: Decimal | None = None


class StartRequest(PlanRequest):
    confirmed: bool = False


class PlanView(BaseModel):
    cells: int
    by_reason: dict[str, int]
    sample: list[PlannedCell]
    skipped_pinned: int
    estimate: CostEstimate
    cap_usd: Decimal
    needs_confirmation: bool
    confirmation_reason: str | None
    warnings: list[str]


def view(plan: RunPlan) -> PlanView:
    return PlanView(
        cells=len(plan.cells),
        by_reason=plan.by_reason,
        sample=plan.cells[:SAMPLE_CELLS],
        skipped_pinned=plan.skipped_pinned,
        estimate=plan.estimate,
        cap_usd=plan.cap_usd,
        needs_confirmation=plan.needs_confirmation,
        confirmation_reason=plan.confirmation_reason,
        warnings=plan.warnings,
    )


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.post("/workspaces/{workspace_id}/runs/plan", response_model=PlanView)
async def plan_run(workspace_id: UUID, body: PlanRequest, runs: RunsDep) -> PlanView:
    try:
        return view(await runs.plan(workspace_id, body.scope, body.max_cost_usd))
    except repo.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None
    except (ScopeError, InvalidCap) as exc:
        raise _bad_request(exc) from None


@router.post(
    "/workspaces/{workspace_id}/runs",
    response_model=RunRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_run(workspace_id: UUID, body: StartRequest, runs: RunsDep) -> RunRecord:
    try:
        return await runs.start(
            workspace_id,
            body.scope,
            confirmed=body.confirmed,
            cap_usd=body.max_cost_usd,
        )
    except ConfirmationRequired as needed:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"this run needs a yes first: {needed.plan.confirmation_reason}",
                "plan": view(needed.plan).model_dump(mode="json"),
            },
        ) from None
    except RunInProgressError:
        raise HTTPException(
            status_code=409, detail="a run is already going in this workspace"
        ) from None
    except repo.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None
    except (ScopeError, InvalidCap) as exc:
        raise _bad_request(exc) from None


@router.get("/workspaces/{workspace_id}/runs", response_model=list[RunRecord])
async def list_runs(workspace_id: UUID, runs: RunsDep) -> list[RunRecord]:
    return await runs.recent(workspace_id)


@router.get("/workspaces/{workspace_id}/runs/{run_id}", response_model=RunRecord)
async def get_run(workspace_id: UUID, run_id: UUID, runs: RunsDep) -> RunRecord:
    record = await runs.get(workspace_id, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return record


@router.post(
    "/workspaces/{workspace_id}/runs/{run_id}/stop",
    response_model=RunRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
async def stop_run(workspace_id: UUID, run_id: UUID, runs: RunsDep) -> RunRecord:
    try:
        return await runs.stop(workspace_id, run_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="run not found") from None
