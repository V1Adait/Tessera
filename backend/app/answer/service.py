"""Answering a question from the cells a confirmed plan filled.

    plan confirmed -> columns created, cells filled -> prepare_answer -> a chat turn whose prompt
    holds the question and the table of cells -> the model answers from the table, citing cells

The prompt is sent to the model but not saved as a user message: the person never typed it. The
model's answer is saved like any other assistant message.
"""

from uuid import UUID

import asyncpg

from backend.app import chat_service
from backend.app.answer.cells import DATA_TAGS, build_cell_table
from backend.app.db import agent_repository as arepo
from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.db import runs_repository as rrepo
from backend.app.query.llm_json import neutralise


class ActionMissingError(LookupError):
    """No such plan."""


class NotReadyError(Exception):
    """The plan exists but there is nothing to answer from yet. The message says why."""


async def prepare_answer(
    pool: asyncpg.Pool, action_id: UUID, request_id: UUID
) -> chat_service.Turn:
    action = await arepo.get_action(pool, action_id)
    if action is None or action.tool != "propose_columns":
        raise ActionMissingError(str(action_id))
    if action.status != "confirmed":
        raise NotReadyError(
            "this plan was not confirmed, or it failed: there is nothing to answer from"
        )
    if await rrepo.running_run(pool, action.workspace_id) is not None:
        raise NotReadyError("the table is still being filled")

    planned = [
        c["name"]
        for c in action.args.get("columns", [])
        if isinstance(c, dict) and isinstance(c.get("name"), str)
    ]
    wanted = {name.casefold() for name in planned}
    columns = [
        c
        for c in await mrepo.list_columns(pool, action.workspace_id)
        if c.name.casefold() in wanted
    ]
    if not columns:
        raise NotReadyError("the plan's columns no longer exist")
    rows = await repo.list_entities(pool, action.workspace_id)
    column_ids = {c.id for c in columns}
    cells = [
        c
        for c in await mrepo.list_cells(pool, action.workspace_id)
        if c.column_id in column_ids
    ]
    filenames = await repo.document_filenames(
        pool, list({e.document_id for c in cells for e in c.evidence})
    )

    question = str(action.args.get("question") or "").strip()
    message = (
        f"Question: {neutralise(question, DATA_TAGS)}\n\n"
        f"{build_cell_table(rows, columns, cells, filenames)}\n\n"
        "Answer the question from the cells above."
    )
    return await chat_service.begin_followup_turn(
        pool, action.conversation_id, request_id, message
    )
