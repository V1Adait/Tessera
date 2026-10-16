"""Columns, cells and evidence in the database: versions, the guards, and the rules the schema
enforces by itself. Each test uses a throwaway schema (conftest) and skips without a database."""

from uuid import uuid4

import asyncpg
import pytest

from backend.app.db import matrix_repository as mrepo
from backend.app.db import repository as repo
from backend.app.matrix.models import (
    CellOutcome,
    ColumnDraft,
    ColumnPatch,
    VerifiedEvidence,
)

SHA = "b" * 64


def draft(name="Employees", **kw) -> ColumnDraft:
    return ColumnDraft(
        name=name,
        prompt=kw.pop("prompt", "How many?"),
        type=kw.pop("type", "number"),
        **kw,
    )


@pytest.fixture
async def ws(pool):
    return await repo.default_workspace_id(pool)


@pytest.fixture
async def acme(pool, ws):
    return await repo.create_entity(pool, ws, "Acme")


@pytest.fixture
async def document(pool, ws, acme):
    return await repo.create_document(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        filename="a.docx",
        kind="docx",
        sha256=SHA,
        size_bytes=10,
    )


def evidence(document, quote="approximately 166,000 employees") -> VerifiedEvidence:
    return VerifiedEvidence(
        document_id=document.id,
        chunk_id=None,
        quote=quote,
        char_start=10,
        char_end=40,
        exact_range=True,
        page_start=3,
        page_end=3,
    )


def answered(document, **kw) -> CellOutcome:
    return CellOutcome(
        status="answered",
        value={"number": "166000", "unit": None},
        display="166,000",
        value_type="number",
        evidence=[evidence(document)],
        **kw,
    )


# --- columns ----------------------------------------------------------------------------------


async def test_a_new_column_starts_at_version_one_and_lists_in_order(pool, ws):
    a = await mrepo.create_column(pool, ws, draft("Alpha"))
    b = await mrepo.create_column(pool, ws, draft("Beta"))

    assert (a.version, b.version) == (1, 1)
    assert [c.name for c in await mrepo.list_columns(pool, ws)] == ["Alpha", "Beta"]
    assert (await mrepo.get_column(pool, a.id)) == a


async def test_column_names_are_unique_ignoring_case_and_spacing(pool, ws):
    await mrepo.create_column(pool, ws, draft("Net Sales"))

    with pytest.raises(mrepo.DuplicateColumnError):
        await mrepo.create_column(pool, ws, draft("  net   SALES "))


async def test_the_same_name_may_exist_in_another_workspace(pool, ws):
    other = await repo.create_workspace(pool, "Other")
    await mrepo.create_column(pool, ws, draft("Net Sales"))

    assert (
        await mrepo.create_column(pool, other.id, draft("Net Sales"))
    ).workspace_id == other.id


async def test_a_column_in_an_unknown_workspace_is_refused(pool):
    with pytest.raises(repo.WorkspaceNotFoundError):
        await mrepo.create_column(pool, uuid4(), draft())


async def test_a_column_can_only_depend_on_columns_of_its_own_workspace(pool, ws):
    other = await repo.create_workspace(pool, "Other")
    foreign = await mrepo.create_column(pool, other.id, draft("Foreign"))
    own = await mrepo.create_column(pool, ws, draft("Own"))

    with pytest.raises(mrepo.ColumnNotFoundError):
        await mrepo.create_column(pool, ws, draft("Derived", depends_on=[foreign.id]))
    with pytest.raises(mrepo.ColumnNotFoundError):
        await mrepo.create_column(pool, ws, draft("Derived", depends_on=[uuid4()]))
    ok = await mrepo.create_column(pool, ws, draft("Derived", depends_on=[own.id]))
    assert ok.depends_on == [own.id]


async def test_the_schema_itself_refuses_a_select_with_one_value(pool, ws):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            """
            INSERT INTO column_versions (column_id, version, name, prompt, type, allowed_values)
            VALUES ($1, 2, 'x', 'y', 'select', ARRAY['only'])
            """,
            column.id,
        )


async def test_changing_a_column_adds_a_version_and_keeps_the_old_one(pool, ws):
    column = await mrepo.create_column(pool, ws, draft(prompt="Old question"))

    changed = await mrepo.update_column(
        pool, column.id, ColumnPatch(prompt="New question")
    )

    assert (changed.version, changed.prompt) == (2, "New question")
    rows = await pool.fetch(
        "select version, prompt from column_versions where column_id = $1 order by version",
        column.id,
    )
    assert [(r["version"], r["prompt"]) for r in rows] == [
        (1, "Old question"),
        (2, "New question"),
    ]


async def test_a_patch_that_changes_nothing_makes_no_version(pool, ws):
    column = await mrepo.create_column(pool, ws, draft())

    same = await mrepo.update_column(
        pool, column.id, ColumnPatch(name=column.name, prompt=column.prompt)
    )

    assert same.version == 1
    assert await pool.fetchval("select count(*) from column_versions") == 1


async def test_a_patch_leaves_unmentioned_fields_alone_and_can_clear_the_applicability_rule(
    pool, ws
):
    column = await mrepo.create_column(
        pool, ws, draft(applicability="Only listed companies", evidence_required=False)
    )

    renamed = await mrepo.update_column(pool, column.id, ColumnPatch(name="Headcount"))
    cleared = await mrepo.update_column(
        pool, column.id, ColumnPatch(clear_applicability=True)
    )

    assert (
        renamed.applicability == "Only listed companies"
        and renamed.evidence_required is False
    )
    assert cleared.applicability is None and cleared.version == 3


async def test_a_rename_to_an_existing_name_is_refused_and_changes_nothing(pool, ws):
    await mrepo.create_column(pool, ws, draft("Taken"))
    column = await mrepo.create_column(pool, ws, draft("Free"))

    with pytest.raises(mrepo.DuplicateColumnError):
        await mrepo.update_column(pool, column.id, ColumnPatch(name="taken"))

    assert (await mrepo.get_column(pool, column.id)).version == 1  # type: ignore[union-attr]
    assert (
        await pool.fetchval(
            "select count(*) from column_versions where column_id = $1", column.id
        )
        == 1
    )


async def test_an_invalid_patch_changes_nothing(pool, ws):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(ValueError):
        await mrepo.update_column(
            pool, column.id, ColumnPatch(type="select", allowed_values=["only"])
        )

    assert (await mrepo.get_column(pool, column.id)).type == "number"  # type: ignore[union-attr]


async def test_moving_a_select_column_to_another_type_drops_its_value_list(pool, ws):
    column = await mrepo.create_column(
        pool, ws, draft("Kind", type="select", allowed_values=["A", "B"])
    )

    changed = await mrepo.update_column(pool, column.id, ColumnPatch(type="text"))

    assert changed.type == "text" and changed.allowed_values == []


async def test_a_column_cannot_depend_on_itself_and_an_unknown_column_is_refused(
    pool, ws
):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(ValueError, match="itself"):
        await mrepo.update_column(pool, column.id, ColumnPatch(depends_on=[column.id]))
    with pytest.raises(mrepo.ColumnNotFoundError):
        await mrepo.update_column(pool, uuid4(), ColumnPatch(name="x"))


# --- cells and evidence -----------------------------------------------------------------------


async def test_a_saved_cell_comes_back_with_its_evidence_in_order(
    pool, ws, acme, document
):
    column = await mrepo.create_column(pool, ws, draft())
    outcome = answered(document)
    outcome.evidence.append(evidence(document, quote="a second quote"))

    cell = await mrepo.save_cell(
        pool, workspace_id=ws, entity_id=acme.id, column=column, outcome=outcome
    )

    assert [e.quote for e in cell.evidence] == [
        "approximately 166,000 employees",
        "a second quote",
    ]
    assert [e.seq for e in cell.evidence] == [0, 1]
    assert cell.evidence[0].page_start == 3 and cell.evidence[0].exact_range is True


async def test_an_answered_cell_without_evidence_is_refused_when_the_column_requires_it(
    pool, ws, acme
):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(mrepo.MissingEvidenceError):
        await mrepo.save_cell(
            pool,
            workspace_id=ws,
            entity_id=acme.id,
            column=column,
            outcome=CellOutcome(status="answered", value=5, display="5"),
        )

    assert await pool.fetchval("select count(*) from cells") == 0


async def test_an_answered_cell_without_evidence_is_allowed_when_the_column_does_not_require_it(
    pool, ws, acme
):
    column = await mrepo.create_column(pool, ws, draft(evidence_required=False))

    cell = await mrepo.save_cell(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        column=column,
        outcome=CellOutcome(status="answered", value=5, display="5"),
    )

    assert cell.evidence == []


async def test_the_schema_itself_refuses_an_answered_cell_with_no_value(pool, ws, acme):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "INSERT INTO cells (workspace_id, entity_id, column_id, column_version, status) VALUES ($1, $2, $3, 1, 'answered')",
            ws,
            acme.id,
            column.id,
        )


async def test_the_schema_itself_allows_one_cell_per_row_and_column(pool, ws, acme):
    column = await mrepo.create_column(pool, ws, draft())
    sql = "INSERT INTO cells (workspace_id, entity_id, column_id, column_version, status) VALUES ($1, $2, $3, 1, 'not_found')"
    await pool.execute(sql, ws, acme.id, column.id)

    with pytest.raises(asyncpg.UniqueViolationError):
        await pool.execute(sql, ws, acme.id, column.id)


async def test_the_schema_itself_refuses_an_unknown_status(pool, ws, acme):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "INSERT INTO cells (workspace_id, entity_id, column_id, column_version, status) VALUES ($1, $2, $3, 1, 'maybe')",
            ws,
            acme.id,
            column.id,
        )


async def test_a_cell_cannot_join_a_row_and_a_column_from_different_workspaces(
    pool, ws, acme
):
    other = await repo.create_workspace(pool, "Other")
    foreign_column = await mrepo.create_column(pool, other.id, draft("Foreign"))

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await pool.execute(
            "INSERT INTO cells (workspace_id, entity_id, column_id, column_version, status) VALUES ($1, $2, $3, 1, 'not_found')",
            ws,
            acme.id,
            foreign_column.id,
        )


async def test_a_cell_cannot_name_a_column_version_that_does_not_exist(pool, ws, acme):
    column = await mrepo.create_column(pool, ws, draft())

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await pool.execute(
            "INSERT INTO cells (workspace_id, entity_id, column_id, column_version, status) VALUES ($1, $2, $3, 9, 'not_found')",
            ws,
            acme.id,
            column.id,
        )


async def test_a_failed_save_leaves_no_half_written_cell(pool, ws, acme, document):
    column = await mrepo.create_column(pool, ws, draft())
    bad = answered(document)
    bad.evidence[0] = bad.evidence[0].model_copy(
        update={"document_id": uuid4()}
    )  # no such document

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await mrepo.save_cell(
            pool, workspace_id=ws, entity_id=acme.id, column=column, outcome=bad
        )

    assert await pool.fetchval("select count(*) from cells") == 0
    assert await pool.fetchval("select count(*) from cell_evidence") == 0


async def test_saving_again_replaces_the_evidence(pool, ws, acme, document):
    column = await mrepo.create_column(pool, ws, draft())
    await mrepo.save_cell(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        column=column,
        outcome=answered(document),
    )
    second = answered(document)
    second.evidence = [evidence(document, quote="a different quote")]

    cell = await mrepo.save_cell(
        pool, workspace_id=ws, entity_id=acme.id, column=column, outcome=second
    )

    assert [e.quote for e in cell.evidence] == ["a different quote"]
    assert await pool.fetchval("select count(*) from cell_evidence") == 1


async def test_deleting_a_row_deletes_its_cells_and_evidence(pool, ws, acme, document):
    column = await mrepo.create_column(pool, ws, draft())
    await mrepo.save_cell(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        column=column,
        outcome=answered(document),
    )

    await pool.execute("DELETE FROM entities WHERE id = $1", acme.id)

    assert await pool.fetchval("select count(*) from cells") == 0
    assert await pool.fetchval("select count(*) from cell_evidence") == 0


async def test_deleting_a_column_deletes_its_cells(pool, ws, acme, document):
    column = await mrepo.create_column(pool, ws, draft())
    await mrepo.save_cell(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        column=column,
        outcome=answered(document),
    )

    await pool.execute("DELETE FROM matrix_columns WHERE id = $1", column.id)

    assert await pool.fetchval("select count(*) from cells") == 0


async def test_deleting_a_document_deletes_the_evidence_taken_from_it_but_not_the_cell(
    pool, ws, acme, document
):
    column = await mrepo.create_column(pool, ws, draft())
    cell = await mrepo.save_cell(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        column=column,
        outcome=answered(document),
    )

    await pool.execute("DELETE FROM documents WHERE id = $1", document.id)

    kept = await mrepo.get_cell(pool, cell.id)
    assert (
        kept is not None and kept.evidence == []
    )  # now an answered cell with no evidence: visible to a check


async def test_pinning_and_releasing_a_cell(pool, ws, acme):
    column = await mrepo.create_column(pool, ws, draft())
    cell = await mrepo.save_cell(
        pool,
        workspace_id=ws,
        entity_id=acme.id,
        column=column,
        outcome=CellOutcome(status="not_found", detail="x"),
    )

    assert await mrepo.set_pinned(pool, cell.id, True) is True
    assert (await mrepo.get_cell(pool, cell.id)).pinned is True  # type: ignore[union-attr]
    assert await mrepo.set_pinned(pool, uuid4(), True) is False


async def test_listing_cells_can_filter_by_row_and_column(pool, ws, acme):
    beta = await repo.create_entity(pool, ws, "Beta")
    c1 = await mrepo.create_column(pool, ws, draft("One"))
    c2 = await mrepo.create_column(pool, ws, draft("Two"))
    for entity in (acme, beta):
        for column in (c1, c2):
            await mrepo.save_cell(
                pool,
                workspace_id=ws,
                entity_id=entity.id,
                column=column,
                outcome=CellOutcome(status="not_found", detail="x"),
            )

    assert len(await mrepo.list_cells(pool, ws)) == 4
    assert len(await mrepo.list_cells(pool, ws, entity_id=acme.id)) == 2
    assert len(await mrepo.list_cells(pool, ws, column_id=c2.id)) == 2
    assert (
        len(await mrepo.list_cells(pool, ws, entity_id=beta.id, column_id=c1.id)) == 1
    )
    other = await repo.create_workspace(pool, "Other")
    assert await mrepo.list_cells(pool, other.id) == []
