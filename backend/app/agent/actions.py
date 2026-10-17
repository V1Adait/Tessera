"""Confirming or cancelling an action the agent was not allowed to take on its own.

The agent only STORED the call (see loop.handle_call). Here a person says yes or no:

    confirm:  take the action (atomically, so two clicks run it once), check it is not stale,
              run the stored tool call in code, record the result, and tell the conversation.
    cancel:   mark it cancelled and tell the conversation.

The model is not involved. What the person saw in the confirmation is exactly what runs: the same
validated arguments, executed by the same tool.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
from pydantic import BaseModel

from backend.app.agent.tools import (
    Invalid,
    Registry,
    Services,
    ToolContext,
    parse_args,
    run_tool,
)
from backend.app.config import Settings
from backend.app.db import agent_repository as arepo
from backend.app.db import repository as repo

logger = logging.getLogger(__name__)


class ActionNotFoundError(LookupError):
    pass


class ActionNotPendingError(Exception):
    """Already confirmed, cancelled or failed: it cannot run again."""


class ActionExpiredError(Exception):
    """Too old: the workspace may have changed since it was shown. Ask again."""


class ActionOutcome(BaseModel):
    action: arepo.ActionRecord
    ok: bool
    message: str  # what happened, in plain words
    changed_matrix: bool = False
    data: dict[
        str, Any
    ] = {}  # for the screen: for a column plan, the id of the run it started


class ActionDesk:
    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        services: Services,
        registry: Registry,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._services = services
        self._registry = registry

    async def pending(self, conversation_id: UUID) -> list[arepo.ActionRecord]:
        """Actions still waiting. Those past their time limit are not offered."""
        cutoff = datetime.now(tz=UTC) - timedelta(
            minutes=self._settings.agent_action_ttl_minutes
        )
        found = await arepo.list_actions(self._pool, conversation_id, pending_only=True)
        return [a for a in found if a.created_at >= cutoff]

    async def confirm(self, action_id: UUID) -> ActionOutcome:
        action = await arepo.get_action(self._pool, action_id)
        if action is None:
            raise ActionNotFoundError(str(action_id))
        age = datetime.now(tz=UTC) - action.created_at
        if action.status == "pending" and age > timedelta(
            minutes=self._settings.agent_action_ttl_minutes
        ):
            await arepo.cancel_action(self._pool, action_id)
            raise ActionExpiredError(str(action_id))
        claimed = await arepo.claim_action(self._pool, action_id)
        if claimed is None:
            raise ActionNotPendingError(action.status)

        spec = self._registry.get(claimed.tool)
        ctx = ToolContext(
            pool=self._pool,
            settings=self._settings,
            services=self._services,
            workspace_id=claimed.workspace_id,
            conversation_id=claimed.conversation_id,
            confirmed=True,  # a person said yes: tools that insist on this may now run
        )
        if spec is None:
            return await self._finish(
                claimed, False, f"The tool '{claimed.tool}' no longer exists.", False
            )
        try:
            args = parse_args(spec, claimed.args)
        except Invalid as exc:
            return await self._finish(
                claimed, False, f"The saved action is no longer valid: {exc}", False
            )
        result = await run_tool(spec, ctx, args, self._settings.agent_tool_timeout_s)
        return await self._finish(
            claimed, result.ok, result.text, result.changed_matrix, result.data
        )

    async def cancel(self, action_id: UUID) -> ActionOutcome:
        action = await arepo.get_action(self._pool, action_id)
        if action is None:
            raise ActionNotFoundError(str(action_id))
        cancelled = await arepo.cancel_action(self._pool, action_id)
        if cancelled is None:
            raise ActionNotPendingError(action.status)
        await self._tell(
            cancelled, f"Cancelled, nothing was changed: {cancelled.summary}"
        )
        return ActionOutcome(
            action=cancelled, ok=True, message="Cancelled. Nothing was changed."
        )

    async def _finish(
        self,
        action: arepo.ActionRecord,
        ok: bool,
        text: str,
        changed: bool,
        data: dict[str, Any] | None = None,
    ) -> ActionOutcome:
        await arepo.finish_action(self._pool, action.id, ok=ok, result=text)
        lead = "Done" if ok else "Could not do it"
        await self._tell(action, f"{lead}: {action.summary}\n{text}")
        done = await arepo.get_action(self._pool, action.id)
        return ActionOutcome(
            action=done or action,
            ok=ok,
            message=text,
            changed_matrix=changed and ok,
            data=data or {},
        )

    async def _tell(self, action: arepo.ActionRecord, text: str) -> None:
        """Put the outcome in the conversation, so the next message (and the model reading the
        history) knows what happened. A failure to write it is logged, not raised."""
        try:
            await repo.add_message(
                self._pool, action.conversation_id, "assistant", text, request_id=None
            )
        except Exception:
            logger.exception("could not record an action's outcome in the conversation")
