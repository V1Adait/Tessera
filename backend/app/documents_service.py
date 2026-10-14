"""Uploading a document and parsing it in the background.

    upload   check the workspace and entity -> stream the file to disk (intake checks, hashing) ->
             record it -> return at once. The same bytes already in the workspace are not parsed
             again: the existing document comes back, marked as a duplicate.
    parse    claim the document (so two workers never parse it at once) -> run the parser in a
             worker thread with a timeout -> save blocks and status in one transaction.

Failures never raise out of the background job: the document is marked 'failed' with a plain
reason (never internal details), and can be re-parsed.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from backend.app.chunking.service import document_context, make_chunks, rechunk
from backend.app.config import Settings
from backend.app.db import repository as repo
from backend.app.db.records import DocumentRecord
from backend.app.embeddings.indexer import Indexer
from backend.app.ingest.intake import IntakeLimits
from backend.app.ingest.models import ParseError
from backend.app.ingest.parse import build_report, decide_status, page_count, parse_file
from backend.app.ingest.parse_pdf import ScanParser
from backend.app.ingest.storage import path_for, save_upload

logger = logging.getLogger(__name__)


class ParseBusyError(Exception):
    """The document is being parsed right now."""


@dataclass(frozen=True)
class UploadResult:
    document: DocumentRecord
    duplicate: bool  # True: these bytes were already uploaded to this workspace


class DocumentService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        scan_parser: ScanParser | None = None,
        indexer: Indexer | None = None,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.scan_parser = scan_parser
        self.indexer = (
            indexer  # when set, parsed documents are embedded in the background
        )
        self._slots = asyncio.Semaphore(max(1, settings.max_concurrent_parses))
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def upload_dir(self) -> Path:
        return Path(self.settings.upload_dir)

    # --- background plumbing --------------------------------------------------------------

    def _spawn(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        """Run in the background, holding a reference so the task is not garbage-collected."""
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self, timeout: float = 10.0) -> None:
        """At shutdown: give running parses a moment to finish, then cancel the rest."""
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # --- upload ---------------------------------------------------------------------------

    async def upload(
        self,
        *,
        workspace_id: UUID | None,
        entity_id: UUID | None,
        filename: str | None,
        read: Callable[[int], Awaitable[bytes]],
    ) -> UploadResult:
        """Accept a file. Raises IntakeError (refused file), WorkspaceNotFoundError or
        EntityNotInWorkspaceError (checked BEFORE the file is stored, so a bad request leaves
        nothing behind)."""
        workspace = workspace_id or await repo.default_workspace_id(self.pool)
        if await repo.get_workspace(self.pool, workspace) is None:
            raise repo.WorkspaceNotFoundError(str(workspace))
        if entity_id is not None and not await repo.entity_in_workspace(
            self.pool, entity_id, workspace
        ):
            raise repo.EntityNotInWorkspaceError(str(entity_id))

        stored = await save_upload(
            read, filename, self.upload_dir, IntakeLimits.from_settings(self.settings)
        )
        try:
            document = await repo.create_document(
                self.pool,
                workspace_id=workspace,
                entity_id=entity_id,
                filename=stored.display_name,
                kind=stored.kind,
                sha256=stored.sha256,
                size_bytes=stored.size,
            )
        except repo.DuplicateDocumentError as duplicate:
            return UploadResult(duplicate.existing, duplicate=True)
        self._spawn(self.parse(document.id))
        return UploadResult(document, duplicate=False)

    # --- parsing --------------------------------------------------------------------------

    async def parse(self, document_id: UUID) -> None:
        """Claim a document and parse it. Does nothing if another worker already has it."""
        document = await repo.claim_for_parsing(self.pool, document_id)
        if document is not None:
            await self._parse_claimed(document)

    async def reparse(self, document_id: UUID) -> DocumentRecord:
        """Start parsing again (replacing the old blocks when it finishes). Raises LookupError if
        there is no such document and ParseBusyError if it is being parsed right now."""
        document = await repo.claim_for_parsing(self.pool, document_id)
        if document is None:
            if await repo.get_document(self.pool, document_id) is None:
                raise LookupError(f"document {document_id} does not exist")
            raise ParseBusyError(str(document_id))
        self._spawn(self._parse_claimed(document))
        return document

    async def assign_entity(self, document_id: UUID, entity_id: UUID | None) -> None:
        """Attach a document to an entity (or detach it). The chunks' search text names the
        entity, so they are rebuilt from the stored blocks. Raises what repo.assign_entity does."""
        await repo.assign_entity(self.pool, document_id, entity_id)
        document = await repo.get_document(self.pool, document_id)
        if document is not None and document.status in ("parsed", "needs_review"):
            await rechunk(self.pool, self.settings, document_id)
            if self.indexer is not None:
                self.indexer.schedule(document_id)

    async def _fail(self, document: DocumentRecord, reason: str) -> None:
        try:
            await repo.set_document_status(self.pool, document.id, "failed", reason)
        except Exception:
            logger.exception("could not mark document %s as failed", document.id)

    async def _parse_claimed(self, document: DocumentRecord) -> None:
        """Parse a document this worker has claimed. Never raises."""
        path = path_for(self.upload_dir, document.sha256)
        timeout = self.settings.parse_timeout_s
        try:
            async with self._slots:
                parsed = await asyncio.wait_for(
                    asyncio.to_thread(
                        parse_file, path, document.kind, self.settings, self.scan_parser
                    ),
                    timeout,
                )
            status, detail = decide_status(parsed)
            chunks = await make_chunks(
                parsed.blocks,
                await document_context(self.pool, document),
                self.settings,
            )
            await repo.save_parse_result(
                self.pool,
                document.id,
                parser=parsed.parser,
                status=status,
                detail=detail,
                page_count=page_count(parsed),
                char_count=sum(len(b.text) for b in parsed.blocks),
                report=build_report(parsed),
                blocks=parsed.blocks,
                chunks=chunks,
            )
            if self.indexer is not None:
                self.indexer.schedule(document.id)
        except TimeoutError:
            # the worker thread cannot be killed; it finishes in the background and is ignored
            await self._fail(document, f"parsing took longer than {timeout:g} seconds")
        except ParseError as exc:
            await self._fail(document, exc.message)
        except FileNotFoundError:
            await self._fail(document, "the stored file is missing")
        except asyncio.CancelledError:
            await self._fail(document, "interrupted by a shutdown: use re-parse")
            raise
        except Exception:
            logger.exception("unexpected error parsing document %s", document.id)
            await self._fail(document, "unexpected error while reading the file")
