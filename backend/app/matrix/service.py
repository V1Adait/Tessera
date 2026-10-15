"""The matrix, as the rest of the app sees it: define columns, fill a cell, read cells back.

The extractor decides what a cell says; this layer checks the request makes sense (the column and
the row belong to the workspace, a person's pinned cell is not overwritten by accident) and
stores the result with its evidence in one transaction.
"""

import logging
from uuid import UUID

import asyncpg

from backend.app.config import Settings
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db.records import ChunkHit
from backend.app.embeddings.indexer import Indexer
from backend.app.llm import get_llm
from backend.app.matrix.extractor import CellExtractor, Retriever
from backend.app.matrix.models import CellRecord, ColumnDraft, ColumnPatch, ColumnRecord
from backend.app.retrieval.terms import or_query
from backend.app.tracing import Recorder, db_recorder

logger = logging.getLogger(__name__)


class EntityNotFoundError(LookupError):
    """No such row in this workspace."""


class MatrixService:
    def __init__(
        self, pool: asyncpg.Pool, settings: Settings, extractor: CellExtractor
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._extractor = extractor

    # --- columns ---------------------------------------------------------------------------

    async def create_column(
        self, workspace_id: UUID, draft: ColumnDraft
    ) -> ColumnRecord:
        return await mrepo.create_column(self._pool, workspace_id, draft)

    async def update_column(
        self, workspace_id: UUID, column_id: UUID, patch: ColumnPatch
    ) -> ColumnRecord:
        await self._column_in(workspace_id, column_id)
        return await mrepo.update_column(self._pool, column_id, patch)

    async def list_columns(self, workspace_id: UUID) -> list[ColumnRecord]:
        if await repo.get_workspace(self._pool, workspace_id) is None:
            raise repo.WorkspaceNotFoundError(str(workspace_id))
        return await mrepo.list_columns(self._pool, workspace_id)

    async def _column_in(self, workspace_id: UUID, column_id: UUID) -> ColumnRecord:
        column = await mrepo.get_column(self._pool, column_id)
        if column is None or column.workspace_id != workspace_id:
            raise mrepo.ColumnNotFoundError(str(column_id))
        return column

    # --- cells -----------------------------------------------------------------------------

    async def extract_cell(
        self,
        workspace_id: UUID,
        entity_id: UUID,
        column_id: UUID,
        *,
        replace_pinned: bool = False,
    ) -> CellRecord:
        """Fill (or refill) one cell. Raises ColumnNotFoundError, EntityNotFoundError or
        CellPinnedError; a model or retrieval problem does not raise, it becomes the cell's status."""
        column = await self._column_in(workspace_id, column_id)
        if not await repo.entity_in_workspace(self._pool, entity_id, workspace_id):
            raise EntityNotFoundError(str(entity_id))
        existing = await mrepo.get_cell_for(self._pool, entity_id, column_id)
        if existing is not None and existing.pinned and not replace_pinned:
            raise mrepo.CellPinnedError(
                str(existing.id)
            )  # before any model call is paid for

        entity_name = await repo.entity_name(self._pool, entity_id) or "the company"
        outcome = await self._extractor.extract(
            workspace_id, entity_id, entity_name, column
        )
        return await mrepo.save_cell(
            self._pool,
            workspace_id=workspace_id,
            entity_id=entity_id,
            column=column,
            outcome=outcome,
            replace_pinned=replace_pinned,
        )

    async def get_cell(self, workspace_id: UUID, cell_id: UUID) -> CellRecord | None:
        cell = await mrepo.get_cell(self._pool, cell_id)
        return cell if cell is not None and cell.workspace_id == workspace_id else None

    async def list_cells(
        self,
        workspace_id: UUID,
        *,
        entity_id: UUID | None = None,
        column_id: UUID | None = None,
    ) -> list[CellRecord]:
        if await repo.get_workspace(self._pool, workspace_id) is None:
            raise repo.WorkspaceNotFoundError(str(workspace_id))
        return await mrepo.list_cells(
            self._pool, workspace_id, entity_id=entity_id, column_id=column_id
        )


def make_retriever(
    pool: asyncpg.Pool, indexer: Indexer | None, settings: Settings
) -> Retriever:
    """Search one row's documents: hybrid (meaning + keywords) if the vector index exists, else the
    database's full-text search. Always limited to the row, in the search itself."""

    async def retrieve(
        workspace_id: UUID, query: str, entity_id: UUID, limit: int
    ) -> list[ChunkHit]:
        if indexer is not None:
            result = await indexer.hybrid_search(
                workspace_id, query, entity_ids=[entity_id], limit=limit
            )
            if result.degraded:
                logger.warning("extraction retrieval degraded: %s", result.degraded)
            return result.hits
        terms = or_query(query)
        if not terms:
            return []
        return await repo.search_chunks_text(
            pool, workspace_id, terms, entity_ids=[entity_id], limit=limit
        )

    return retrieve


def build_matrix_service(
    pool: asyncpg.Pool,
    settings: Settings,
    indexer: Indexer | None,
    recorder: Recorder | None = None,
) -> MatrixService | None:
    """The real service, or None if there is no key for the extraction model."""
    provider, model = settings.extract_provider, settings.extract_model
    if not settings.key_for(provider):
        logger.warning("no key for the extraction model: filling cells is off")
        return None
    llm = get_llm(
        settings,
        provider=provider,
        model=model,
        max_tokens=settings.extract_max_tokens,
        temperature=0,
        seed=settings.extract_seed,
    )
    extractor = CellExtractor(
        llm,
        pool=pool,
        settings=settings,
        retrieve=make_retriever(pool, indexer, settings),
        provider=provider,
        model=model,
        recorder=recorder or db_recorder(pool),
    )
    return MatrixService(pool, settings, extractor)
