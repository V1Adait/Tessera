"""The agent, assembled: the thing the chat route hands a turn to.

It holds what lives for the whole server (tools, services, the confirmation desk) and, for each
turn, builds the per-turn pieces (a tool context with its own passage numbering, a budget) and
returns the function the pipeline calls to produce the answer.
"""

from collections.abc import AsyncIterator, Callable
from uuid import UUID

import asyncpg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from backend.app.agent.actions import ActionDesk
from backend.app.agent.loop import AgentDeps, run_agent
from backend.app.agent.prompts import AGENT_RULES
from backend.app.agent.tools import Registry, Services, ToolContext
from backend.app.agent.tools_impl import build_registry
from backend.app.config import Settings
from backend.app.db import agent_repository as arepo
from backend.app.models import StreamEvent
from backend.app.tracing import Recorder, db_recorder

Generator = Callable[[list[BaseMessage]], AsyncIterator[StreamEvent]]


class AgentRuntime:
    system_extra = AGENT_RULES

    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: Settings,
        services: Services,
        recorder: Recorder | None = None,
        registry: Registry | None = None,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._services = services
        self._recorder = recorder or db_recorder(pool)
        self.registry = registry or build_registry()
        self.desk = ActionDesk(pool, settings, services, self.registry)

    def generator(
        self,
        llm: BaseChatModel,
        *,
        workspace_id: UUID,
        conversation_id: UUID,
        request_id: UUID,
    ) -> Generator:
        """The function the pipeline calls with the finished prompt. It produces the turn's events."""
        ctx = ToolContext(
            pool=self._pool,
            settings=self._settings,
            services=self._services,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        deps = AgentDeps(
            llm=llm,
            registry=self.registry,
            ctx=ctx,
            settings=self._settings,
            provider=self._settings.provider or "unknown",
            model=self._settings.default_model,
            request_id=request_id,
            recorder=self._recorder,
        )

        async def generate(messages: list[BaseMessage]) -> AsyncIterator[StreamEvent]:
            spent, _ = await arepo.conversation_spend(self._pool, conversation_id)
            async for event in run_agent(deps, messages, spent):
                yield event

        return generate
