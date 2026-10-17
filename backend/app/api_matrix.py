"""HTTP routes for the matrix: columns and cells. Thin handlers: the work is in matrix/service.py.

What these routes deliberately do NOT do: hand-edit a cell, or fill many cells at once (that is
the runner, with its cost estimate). Deleting a column is here, but the SCREEN asks for the yes
first: the route itself refuses while a fill is going or while computed columns are built on it.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.matrix.models import (
    CellRecord,
    ColumnDraft,
    ColumnPatch,
    ColumnRecord,
    ExtractRequest,
)
from backend.app.matrix.service import (
    ColumnBusyError,
    EntityNotFoundError,
    MatrixService,
)

router = APIRouter(tags=["matrix"])


def get_matrix(request: Request) -> MatrixService:
    service: MatrixService | None = getattr(request.app.state, "matrix", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="the matrix is not configured (needs the database and an extraction model key)",
        )
    return service


MatrixDep = Annotated[MatrixService, Depends(get_matrix)]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} not found")


@router.get("/workspaces/{workspace_id}/columns", response_model=list[ColumnRecord])
async def list_columns(workspace_id: UUID, service: MatrixDep) -> list[ColumnRecord]:
    try:
        return await service.list_columns(workspace_id)
    except repo.WorkspaceNotFoundError:
        raise _not_found("workspace") from None


@router.post(
    "/workspaces/{workspace_id}/columns",
    response_model=ColumnRecord,
    status_code=status.HTTP_201_CREATED,
)
async def create_column(
    workspace_id: UUID, body: ColumnDraft, service: MatrixDep
) -> ColumnRecord:
    try:
        return await service.create_column(workspace_id, body)
    except repo.WorkspaceNotFoundError:
        raise _not_found("workspace") from None
    except mrepo.DuplicateColumnError:
        raise HTTPException(
            status_code=409, detail="a column with this name already exists"
        ) from None
    except mrepo.ColumnNotFoundError:
        raise HTTPException(
            status_code=422, detail="a column this one depends on does not exist here"
        ) from None


@router.patch(
    "/workspaces/{workspace_id}/columns/{column_id}", response_model=ColumnRecord
)
async def update_column(
    workspace_id: UUID, column_id: UUID, body: ColumnPatch, service: MatrixDep
) -> ColumnRecord:
    """Change a column. A real change makes a new version and marks its old answers stale."""
    try:
        return await service.update_column(workspace_id, column_id, body)
    except mrepo.ColumnNotFoundError:
        raise _not_found("column") from None
    except mrepo.DuplicateColumnError:
        raise HTTPException(
            status_code=409, detail="a column with this name already exists"
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.post("/workspaces/{workspace_id}/cells/extract", response_model=CellRecord)
async def extract_cell(
    workspace_id: UUID, body: ExtractRequest, service: MatrixDep
) -> CellRecord:
    """Fill one cell from the row's documents, and return it with its evidence. A model problem
    is not an HTTP error: the cell comes back with status pending_or_error and the reason."""
    try:
        return await service.extract_cell(
            workspace_id,
            body.entity_id,
            body.column_id,
            replace_pinned=body.replace_pinned,
            refresh=body.refresh,
        )
    except mrepo.ColumnNotFoundError:
        raise _not_found("column") from None
    except EntityNotFoundError:
        raise _not_found("row") from None
    except mrepo.CellPinnedError:
        raise HTTPException(
            status_code=409,
            detail="this cell was set by a person; send replace_pinned=true to overwrite it",
        ) from None


@router.get("/workspaces/{workspace_id}/cells", response_model=list[CellRecord])
async def list_cells(
    workspace_id: UUID,
    service: MatrixDep,
    entity_id: UUID | None = None,
    column_id: Annotated[UUID | None, Query()] = None,
) -> list[CellRecord]:
    try:
        return await service.list_cells(
            workspace_id, entity_id=entity_id, column_id=column_id
        )
    except repo.WorkspaceNotFoundError:
        raise _not_found("workspace") from None


@router.get("/workspaces/{workspace_id}/cells/{cell_id}", response_model=CellRecord)
async def get_cell(workspace_id: UUID, cell_id: UUID, service: MatrixDep) -> CellRecord:
    cell = await service.get_cell(workspace_id, cell_id)
    if cell is None:
        raise _not_found("cell")
    return cell


@router.delete(
    "/workspaces/{workspace_id}/columns/{column_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_column(
    workspace_id: UUID, column_id: UUID, service: MatrixDep
) -> None:
    """Remove a column and its cells. 409 (with the reason) while a fill is going or while computed
    columns are built on it. Cannot be undone."""
    try:
        await service.delete_column(workspace_id, column_id)
    except mrepo.ColumnNotFoundError:
        raise _not_found("column") from None
    except ColumnBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
