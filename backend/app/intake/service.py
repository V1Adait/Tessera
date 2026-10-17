"""Intake: after a file is parsed, find (or make) the row it belongs to and attach it.

    parsed file  ->  already attached by a person?  -> leave it alone
                 ->  read its first page (one model call)  -> who is it about?
                 ->  decide in code (match.py): attach to a row / make a row / ask a person
                 ->  record what happened, so the screen can say so

A person's choice always wins: a file that already has a row is never moved. A wrong guess is worse
than a question, so anything unsure is recorded as `needs_review` and waits. Never raises.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

import asyncpg

from backend.app.config import Settings
from backend.app.db import intake_repository as irepo
from backend.app.db import repository as repo
from backend.app.intake.identify import IdentityExtractor
from backend.app.intake.match import Decision, decide, normalise
from backend.app.intake.models import Identity
from backend.app.llm import get_llm
from backend.app.llm_cache import PgLlmCache
from backend.app.tracing import Recorder, db_recorder

logger = logging.getLogger(__name__)

Assign = Callable[[UUID, UUID | None], Awaitable[None]]


class IntakeService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        identifier: IdentityExtractor | None,
        assign: Assign,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._identifier = identifier
        self._assign = (
            assign  # DocumentService.assign_entity: also rebuilds the search text
        )
        self._locks: dict[UUID, asyncio.Lock] = {}

    def _lock(self, workspace_id: UUID) -> asyncio.Lock:
        """One file at a time per workspace: two files about a NEW company must make one row, not two."""
        return self._locks.setdefault(workspace_id, asyncio.Lock())

    async def process(self, document_id: UUID) -> irepo.IntakeRecord | None:
        """Decide for one parsed file. None when there is nothing to do (not parsed, or gone)."""
        document = await repo.get_document(self._pool, document_id)
        if document is None or document.status not in ("parsed", "needs_review"):
            return None
        async with self._lock(document.workspace_id):
            try:
                return await self._decide_and_act(document_id)
            except Exception:
                logger.exception("intake failed for document %s", document_id)
                return await self._save(
                    document_id,
                    "needs_review",
                    None,
                    None,
                    "something went wrong while placing this file: choose its row by hand",
                    [],
                )

    async def _decide_and_act(self, document_id: UUID) -> irepo.IntakeRecord | None:
        document = await repo.get_document(
            self._pool, document_id
        )  # fresh, inside the lock
        if document is None:
            return None
        if document.entity_id is not None:
            existing = await irepo.get_result(self._pool, document_id)
            if existing is not None:
                return existing  # a re-parse must not rewrite what was decided
            return await self._save(
                document_id,
                "skipped",
                document.entity_id,
                None,
                "already attached to a row",
                [],
            )
        if self._identifier is None:
            return await self._save(
                document_id,
                "needs_review",
                None,
                None,
                "no model is configured to read the file",
                [],
            )
        text = await repo.document_text_start(
            self._pool, document_id, self._settings.intake_max_chars
        )
        if not text.strip():
            return await self._save(
                document_id,
                "needs_review",
                None,
                None,
                "the file has no text to read",
                [],
            )
        identity = await self._identifier.identify(document.filename, text)
        entities = await repo.list_entities(self._pool, document.workspace_id)
        decision = decide(
            identity, entities, min_confidence=self._settings.intake_min_confidence
        )
        return await self._act(document.workspace_id, document_id, identity, decision)

    async def _act(
        self,
        workspace_id: UUID,
        document_id: UUID,
        identity: Identity,
        decision: Decision,
    ) -> irepo.IntakeRecord:
        if decision.action == "review":
            return await self._save(
                document_id,
                "needs_review",
                None,
                identity,
                decision.reason,
                decision.candidates,
            )

        if decision.action == "assign":
            assert decision.entity is not None
            entity_id = decision.entity.id
            await repo.learn_entity_identity(
                self._pool,
                entity_id,
                ticker=decision.learn_ticker,
                alias=decision.learn_alias,
                domain=decision.learn_domain,
            )
            await self._assign(document_id, entity_id)
            return await self._save(
                document_id, "assigned", entity_id, identity, decision.reason, []
            )

        assert decision.new_name is not None
        outcome = "created"
        try:
            entity = await repo.create_entity(
                self._pool,
                workspace_id,
                decision.new_name,
                decision.new_ticker,
                decision.new_domain,
            )
        except repo.DuplicateNameError:
            # A row with this name appeared meanwhile (made by hand, say). Use it.
            existing = next(
                (
                    e
                    for e in await repo.list_entities(self._pool, workspace_id)
                    if normalise(e.name) == normalise(decision.new_name)
                ),
                None,
            )
            if existing is None:
                return await self._save(
                    document_id,
                    "needs_review",
                    None,
                    identity,
                    "could not make the row",
                    [],
                )
            entity, outcome = existing, "assigned"
        await self._assign(document_id, entity.id)
        return await self._save(
            document_id, outcome, entity.id, identity, decision.reason, []
        )

    async def _save(
        self,
        document_id: UUID,
        outcome: str,
        entity_id: UUID | None,
        identity: Identity | None,
        detail: str | None,
        candidates: list[str],
    ) -> irepo.IntakeRecord:
        return await irepo.save_result(
            self._pool,
            document_id,
            outcome=outcome,
            entity_id=entity_id,
            detected=identity.model_dump() if identity else {},
            detail=detail,
            candidates=candidates,
        )


def build_intake_service(
    pool: asyncpg.Pool,
    settings: Settings,
    assign: Assign,
    recorder: Recorder | None = None,
) -> IntakeService | None:
    """The real service, or None when intake is switched off. Without a key for the intake model the
    service still runs, but every file waits for a person (and says why)."""
    if not settings.intake_enabled:
        return None
    provider = settings.intake_provider
    identifier: IdentityExtractor | None = None
    if settings.key_for(provider):
        identifier = IdentityExtractor(
            get_llm(
                settings,
                provider=provider,
                model=settings.intake_model,
                max_tokens=settings.intake_max_tokens,
                temperature=0,
            ),
            provider=provider,
            model=settings.intake_model,
            max_chars=settings.intake_max_chars,
            recorder=recorder or db_recorder(pool),
            cache=PgLlmCache(pool) if settings.llm_cache else None,
        )
    else:
        logger.warning(
            "no key for the intake model: files will wait for a person to place them"
        )
    return IntakeService(pool, settings, identifier, assign)
