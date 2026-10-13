"""HTTP route for query handling. A thin handler: the work is in query/service.py."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.app.db import repository as repo
from backend.app.query.models import PlanRequest, QueryPlan
from backend.app.query.service import QueryService

router = APIRouter(tags=["query"])


def get_query_service(request: Request) -> QueryService:
    service: QueryService | None = getattr(request.app.state, "query", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="query planning is not configured (needs the database and a classifier key)",
        )
    return service


QueryDep = Annotated[QueryService, Depends(get_query_service)]


@router.post("/workspaces/{workspace_id}/query/plan", response_model=QueryPlan)
async def plan_query(
    workspace_id: UUID, body: PlanRequest, service: QueryDep
) -> QueryPlan:
    """What would have to be looked up to answer this question? Nothing is retrieved."""
    try:
        return await service.plan(workspace_id, body.question)
    except repo.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None
