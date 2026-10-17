"""The matrix, as the rest of the app sees it: define columns, fill a cell, read cells back.

The extractor decides what a cell says; this layer checks the request makes sense (the column and
the row belong to the workspace, a person's pinned cell is not overwritten by accident) and
stores the result with its evidence in one transaction.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

import asyncpg

from backend.app.config import Settings
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db import runs_repository as rrepo
from backend.app.db.records import ChunkHit
from backend.app.embeddings.indexer import Indexer
from backend.app.finance.checks import CheckSpec
from backend.app.llm import get_llm
from backend.app.llm_cache import PgLlmCache
from backend.app.matrix import computed
from backend.app.matrix.extractor import CellExtractor, Retriever
from backend.app.matrix.models import (
    CellOutcome,
    CellRecord,
    ColumnDraft,
    ColumnPatch,
    ColumnRecord,
)
from backend.app.packs.loader import get_check_specs
from backend.app.retrieval.terms import or_query
from backend.app.tracing import Recorder, db_recorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filled:
    """A cell that was just filled, and what filling it cost."""

    cell: CellRecord
    cost_usd: Decimal
    unknown_cost_calls: int
    model_calls: int


class EntityNotFoundError(LookupError):
    """No such row in this workspace."""


class ColumnBusyError(Exception):
    """The column cannot be removed right now. The message says why, in words for a person."""


class MatrixService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        extractor: CellExtractor,
        check_specs: list[CheckSpec] | None = None,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._extractor = extractor
        # the tie-outs to run on a row's cells; the shipped packs' checks unless a test passes its own
        self._checks = get_check_specs() if check_specs is None else check_specs

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
        refresh: bool = False,
    ) -> CellRecord:
        """Fill (or refill) one cell. Raises ColumnNotFoundError, EntityNotFoundError or
        CellPinnedError; a model or retrieval problem does not raise, it becomes the cell's status."""
        return (
            await self.fill_cell(
                workspace_id,
                entity_id,
                column_id,
                replace_pinned=replace_pinned,
                refresh=refresh,
            )
        ).cell

    async def fill_cell(
        self,
        workspace_id: UUID,
        entity_id: UUID,
        column_id: UUID,
        *,
        replace_pinned: bool = False,
        refresh: bool = False,
    ) -> Filled:
        """Like extract_cell, but also says what the fill cost (a run needs this to keep its cap)."""
        column = await self._column_in(workspace_id, column_id)
        if not await repo.entity_in_workspace(self._pool, entity_id, workspace_id):
            raise EntityNotFoundError(str(entity_id))
        existing = await mrepo.get_cell_for(self._pool, entity_id, column_id)
        if existing is not None and existing.pinned and not replace_pinned:
            raise mrepo.CellPinnedError(
                str(existing.id)
            )  # before any model call is paid for

        if column.derivation is not None:
            # Computed in code from other cells: no model, no cost.
            columns = await mrepo.list_columns(self._pool, workspace_id)
            cell = await computed.compute_cell(
                self._pool,
                workspace_id,
                entity_id,
                column,
                columns,
                replace_pinned=replace_pinned,
            )
            return Filled(
                cell=await self._after_change(workspace_id, entity_id, cell),
                cost_usd=Decimal(0),
                unknown_cost_calls=0,
                model_calls=0,
            )

        entity_name = await repo.entity_name(self._pool, entity_id) or "the company"
        outcome = await self._extractor.extract(
            workspace_id, entity_id, entity_name, column, refresh=refresh
        )
        cell = await mrepo.save_cell(
            self._pool,
            workspace_id=workspace_id,
            entity_id=entity_id,
            column=column,
            outcome=outcome,
            replace_pinned=replace_pinned,
        )
        cell = await self._after_change(workspace_id, entity_id, cell)
        return Filled(
            cell=cell,
            cost_usd=outcome.cost_usd,
            unknown_cost_calls=outcome.unknown_cost_calls,
            model_calls=outcome.model_calls,
        )

    async def delete_column(self, workspace_id: UUID, column_id: UUID) -> None:
        """Remove a column with all its cells. This cannot be undone, so the person has confirmed it
        before this is called (the screen asks; the agent stores it as a pending action).

        Raises ColumnNotFoundError, or ColumnBusyError when a fill is going (it would write cells
        into a column that is being removed) or when computed columns are built on this one."""
        column = await self._column_in(workspace_id, column_id)
        if await rrepo.running_run(self._pool, workspace_id) is not None:
            raise ColumnBusyError(
                "A fill is going in this workspace. Stop it, or wait, before removing a column."
            )
        users = await mrepo.columns_depending_on(self._pool, workspace_id, column_id)
        if users:
            raise ColumnBusyError(
                f"{', '.join(users)} {'is' if len(users) == 1 else 'are'} computed from "
                f"'{column.name}'. Remove or change {'it' if len(users) == 1 else 'them'} first."
            )
        await mrepo.delete_column(self._pool, workspace_id, column_id)
        # tie-outs that named this column no longer apply: bring every row up to date
        entities = await repo.list_entities(self._pool, workspace_id)
        await self.refresh_rows(workspace_id, {e.id for e in entities})

    async def refresh_row(self, workspace_id: UUID, entity_id: UUID) -> None:
        """Bring a row's computed cells and tie-outs up to date with its other cells."""
        await computed.refresh_row(self._pool, workspace_id, entity_id, self._checks)

    async def refresh_rows(self, workspace_id: UUID, entity_ids: set[UUID]) -> None:
        """Refresh several rows. One row failing is logged, not allowed to stop the others."""
        for entity_id in entity_ids:
            try:
                await self.refresh_row(workspace_id, entity_id)
            except Exception:
                logger.exception("could not refresh a row's computed cells")

    async def _after_change(
        self, workspace_id: UUID, entity_id: UUID, cell: CellRecord
    ) -> CellRecord:
        """A cell of this row changed: refresh the computed cells and tie-outs, and return the cell
        as it is now. The refresh never fails a fill: it is derived data, recomputed next time."""
        try:
            await self.refresh_row(workspace_id, entity_id)
            fresh = await mrepo.get_cell(self._pool, cell.id)
            return fresh if fresh is not None else cell
        except Exception:
            logger.exception("could not refresh a row's computed cells")
            return cell

    async def record_failure(
        self, workspace_id: UUID, entity_id: UUID, column_id: UUID, detail: str
    ) -> None:
        """A cell that could not be filled at all (a timeout, an unexpected error) is recorded as
        pending_or_error with a reason, so the grid shows it, not a silent gap. A pinned cell is
        left alone."""
        column = await self._column_in(workspace_id, column_id)
        try:
            await mrepo.save_cell(
                self._pool,
                workspace_id=workspace_id,
                entity_id=entity_id,
                column=column,
                outcome=CellOutcome(status="pending_or_error", detail=detail),
            )
        except mrepo.CellPinnedError:
            pass

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
        cache=PgLlmCache(pool) if settings.llm_cache else None,
    )
    return MatrixService(pool, settings, extractor)
