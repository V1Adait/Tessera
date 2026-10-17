"""Export routes: the matrix as a file to hand to someone.

    GET /workspaces/{id}/export/pdf    a summary table, then the sources behind every figure

Thin handler: the document is built by export/pdf.py (pure), the data comes from the repositories.
"""

import asyncio
import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Response

from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.export.pdf import ExportTooLargeError, build_pdf

router = APIRouter(tags=["export"])

PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]


def _file_name(workspace_name: str, now: datetime) -> str:
    """A safe ASCII file name: letters, digits and dashes only, whatever the workspace is called."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", workspace_name).strip("-").lower() or "matrix"
    return f"tessera-{slug[:60]}-{now:%Y%m%d}.pdf"


@router.get("/workspaces/{workspace_id}/export/pdf")
async def export_pdf(workspace_id: UUID, pool: PoolDep) -> Response:
    """The workspace as a PDF. 404 for an unknown workspace; 422 (with advice) if it is too big."""
    workspace = await repo.get_workspace(pool, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    rows = await repo.list_entities(pool, workspace_id)
    columns = await mrepo.list_columns(pool, workspace_id)
    cells = await mrepo.list_cells(pool, workspace_id)
    documents = await repo.list_documents(pool, workspace_id, limit=200)

    by_row: dict[UUID, list[str]] = {}
    for d in documents:
        if d.entity_id is not None:
            by_row.setdefault(d.entity_id, []).append(d.filename)
    ids = list({e.document_id for c in cells for e in c.evidence})
    filenames = await repo.document_filenames(pool, ids) if ids else {}

    now = datetime.now(tz=UTC)
    try:
        # reportlab is synchronous and CPU-bound: keep it off the event loop
        data = await asyncio.to_thread(
            build_pdf, workspace.name, rows, columns, cells, by_row, filenames, now
        )
    except ExportTooLargeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{_file_name(workspace.name, now)}"',
            "Cache-Control": "no-store",  # it holds the person's data: do not let a cache keep it
        },
    )
