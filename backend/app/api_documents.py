"""HTTP routes for documents: upload, status, parsed blocks, re-parse, assign to an entity.

Thin handlers. The work is in DocumentService (upload and background parsing), the repository
(SQL), and the intake module (what a file really is). Everything the client sends is untrusted:
the file name is only a label, the type comes from the bytes, and limits are enforced on the server.
"""

from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel

from backend.app.config import Settings, get_settings
from backend.app.db import intake_repository as irepo
from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.db.records import BlockKind, BlockRecord, ChunkRecord, DocumentRecord
from backend.app.documents_service import DocumentService, ParseBusyError
from backend.app.ingest.intake import IntakeError
from backend.app.intake.service import IntakeService

router = APIRouter(tags=["documents"])

_ENVELOPE_SLACK = (
    1024 * 1024
)  # a multipart body is a little bigger than the file inside it


def get_documents(request: Request) -> DocumentService:
    service = getattr(request.app.state, "documents", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="document storage is not configured"
        )
    return service  # type: ignore[no-any-return]


ServiceDep = Annotated[DocumentService, Depends(get_documents)]
PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class DocumentView(DocumentRecord):
    block_count: int | None = None  # how many blocks the parser produced
    chunk_count: int | None = None  # how many chunks were built from them
    duplicate: bool = (
        False  # on upload: these bytes were already a document in the workspace
    )


class AssignEntity(BaseModel):
    entity_id: UUID | None  # None un-assigns the document


def _view(document: DocumentRecord, **extra: object) -> DocumentView:
    return DocumentView(**document.model_dump(), **extra)  # type: ignore[arg-type]


@router.post(
    "/documents", response_model=DocumentView, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    request: Request,
    response: Response,
    service: ServiceDep,
    settings: SettingsDep,
    file: UploadFile,
    workspace_id: Annotated[UUID | None, Form()] = None,
    entity_id: Annotated[UUID | None, Form()] = None,
) -> DocumentView:
    """Upload a file. The response comes back at once with the document in status 'uploaded';
    parsing happens in the background, so poll GET /documents/{id} for the result.

    201 for a new document, 200 when the same bytes are already in the workspace."""
    declared = request.headers.get("content-length", "")
    if (
        declared.isdigit()
        and int(declared) > settings.max_upload_bytes + _ENVELOPE_SLACK
    ):
        raise HTTPException(status_code=413, detail="the upload is too large")
    try:
        result = await service.upload(
            workspace_id=workspace_id,
            entity_id=entity_id,
            filename=file.filename,
            read=file.read,
        )
    except IntakeError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from None
    except repo.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None
    except repo.EntityNotInWorkspaceError:
        raise HTTPException(
            status_code=404, detail="entity not found in this workspace"
        ) from None
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return _view(result.document, duplicate=result.duplicate)


@router.get("/documents", response_model=list[DocumentRecord])
async def list_documents(
    pool: PoolDep,
    workspace_id: UUID | None = None,
    entity_id: UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentRecord]:
    try:
        workspace = workspace_id or await repo.default_workspace_id(pool)
    except repo.WorkspaceNotFoundError:
        raise HTTPException(status_code=404, detail="workspace not found") from None
    if await repo.get_workspace(pool, workspace) is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return await repo.list_documents(
        pool, workspace, entity_id=entity_id, limit=limit, offset=offset
    )


async def _require(pool: asyncpg.Pool, document_id: UUID) -> DocumentRecord:
    document = await repo.get_document(pool, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@router.get("/documents/{document_id}", response_model=DocumentView)
async def get_document(document_id: UUID, pool: PoolDep) -> DocumentView:
    document = await _require(pool, document_id)
    return _view(
        document,
        block_count=await repo.count_blocks(pool, document_id),
        chunk_count=await repo.count_chunks(pool, document_id),
    )


@router.get("/documents/{document_id}/blocks", response_model=list[BlockRecord])
async def get_blocks(
    document_id: UUID,
    pool: PoolDep,
    type: BlockKind | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[BlockRecord]:
    """The parsed blocks in reading order, optionally only one type (e.g. type=table)."""
    await _require(pool, document_id)
    return await repo.list_blocks(
        pool, document_id, block_type=type, limit=limit, offset=offset
    )


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkRecord])
async def get_chunks(
    document_id: UUID,
    pool: PoolDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ChunkRecord]:
    """The chunks built from the parsed blocks, in reading order (read-only, for inspection)."""
    await _require(pool, document_id)
    return await repo.list_chunks(pool, document_id, limit=limit, offset=offset)


@router.post(
    "/documents/{document_id}/reparse",
    response_model=DocumentView,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reparse_document(document_id: UUID, service: ServiceDep) -> DocumentView:
    """Parse the stored file again. The old blocks are replaced only when the new parse is saved."""
    try:
        document = await service.reparse(document_id)
    except ParseBusyError:
        raise HTTPException(
            status_code=409, detail="the document is being parsed right now"
        ) from None
    except LookupError:
        raise HTTPException(status_code=404, detail="document not found") from None
    return _view(document)


@router.patch("/documents/{document_id}", response_model=DocumentView)
async def assign_document(
    document_id: UUID, body: AssignEntity, pool: PoolDep, service: ServiceDep
) -> DocumentView:
    """Attach a document to an entity (a row of the matrix) of the same workspace."""
    try:
        await service.assign_entity(document_id, body.entity_id)
    except repo.EntityNotInWorkspaceError:
        raise HTTPException(
            status_code=404, detail="entity not found in this workspace"
        ) from None
    except LookupError:
        raise HTTPException(status_code=404, detail="document not found") from None
    document = await _require(pool, document_id)
    return _view(
        document,
        block_count=await repo.count_blocks(pool, document_id),
        chunk_count=await repo.count_chunks(pool, document_id),
    )


@router.post(
    "/documents/{document_id}/index",
    response_model=DocumentView,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_document(
    document_id: UUID, pool: PoolDep, service: ServiceDep
) -> DocumentView:
    """Queue the document's chunks for the vector index again (after a failure, or a model change)."""
    document = await _require(pool, document_id)
    if service.indexer is None:
        raise HTTPException(status_code=503, detail="vector indexing is not configured")
    if document.status not in ("parsed", "needs_review"):
        raise HTTPException(status_code=409, detail="the document has not been parsed")
    if not await service.indexer.reindex(document_id):
        raise HTTPException(
            status_code=409, detail="the document is being indexed right now"
        )
    return _view(
        await _require(pool, document_id),
        block_count=await repo.count_blocks(pool, document_id),
        chunk_count=await repo.count_chunks(pool, document_id),
    )


class IntakeView(BaseModel):
    """What happened to a file after it was parsed: which row it went to, or why it is waiting."""

    outcome: str  # assigned | created | needs_review | skipped
    entity_id: (
        UUID | None
    )  # the row the file is attached to NOW (a person may have changed it)
    entity_name: str | None
    detected: dict[str, object]  # company, ticker, doc_type, period, confidence
    detail: str | None
    candidates: list[str]


def get_intake(request: Request) -> IntakeService | None:
    return getattr(request.app.state, "intake", None)  # type: ignore[no-any-return]


IntakeDep = Annotated[IntakeService | None, Depends(get_intake)]


async def _intake_view(pool: asyncpg.Pool, document_id: UUID) -> IntakeView | None:
    result = await irepo.get_result(pool, document_id)
    document = await repo.get_document(pool, document_id)
    if result is None or document is None:
        return None
    return IntakeView(
        outcome=result.outcome,
        entity_id=document.entity_id,
        entity_name=await repo.entity_name(pool, document.entity_id),
        detected=result.detected,
        detail=result.detail,
        candidates=result.candidates,
    )


@router.get("/documents/{document_id}/intake", response_model=IntakeView)
async def document_intake(document_id: UUID, pool: PoolDep) -> IntakeView:
    """Which row intake put the file in. 404 until intake has looked at it."""
    await _require(pool, document_id)
    view = await _intake_view(pool, document_id)
    if view is None:
        raise HTTPException(
            status_code=404, detail="intake has not looked at this file yet"
        )
    return view


@router.post("/documents/{document_id}/intake", response_model=IntakeView)
async def rerun_intake(
    document_id: UUID, pool: PoolDep, intake: IntakeDep
) -> IntakeView:
    """Ask intake to place the file again (a file already attached to a row is left alone)."""
    await _require(pool, document_id)
    if intake is None:
        raise HTTPException(status_code=503, detail="intake is not configured")
    await intake.process(document_id)
    view = await _intake_view(pool, document_id)
    if view is None:
        raise HTTPException(status_code=409, detail="the file has not been parsed yet")
    return view
