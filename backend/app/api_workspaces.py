"""HTTP routes for workspaces and their entities (the rows of the matrix).

Thin handlers: validation lives in the request models, SQL in the repository.
"""

from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import EntityRecord, WorkspaceRecord
from backend.app.models import NewEntity, NewWorkspace

router = APIRouter(tags=["workspaces"])

PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]


@router.get("/workspaces", response_model=list[WorkspaceRecord])
async def list_workspaces(pool: PoolDep) -> list[WorkspaceRecord]:
    return await repo.list_workspaces(pool)


@router.post(
    "/workspaces", response_model=WorkspaceRecord, status_code=status.HTTP_201_CREATED
)
async def create_workspace(body: NewWorkspace, pool: PoolDep) -> WorkspaceRecord:
    return await repo.create_workspace(pool, body.name)


@router.get("/workspaces/{workspace_id}/entities", response_model=list[EntityRecord])
async def list_entities(workspace_id: UUID, pool: PoolDep) -> list[EntityRecord]:
    if await repo.get_workspace(pool, workspace_id) is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return await repo.list_entities(pool, workspace_id)


@router.post(
    "/workspaces/{workspace_id}/entities",
    response_model=EntityRecord,
    status_code=status.HTTP_201_CREATED,
)
async def create_entity(
    workspace_id: UUID, body: NewEntity, pool: PoolDep
) -> EntityRecord:
    try:
        return await repo.create_entity(pool, workspace_id, body.name)
    except repo.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None
    except repo.DuplicateNameError:
        raise HTTPException(
            status_code=409, detail="an entity with this name already exists"
        ) from None
