"""The cache for paid model calls: when it answers, when it must not, and that it never costs.

Needs the database (conftest). The model is scripted, so every call is counted exactly.
"""

from decimal import Decimal

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.documents_service import DocumentService
from backend.app.llm_cache import PgLlmCache, cache_key
from backend.app.matrix.models import ColumnPatch
from backend.tests.fakes import make_settings
from backend.tests.test_matrix_flow import World, good

MSGS = [SystemMessage("rules"), HumanMessage("question + passages")]


def key(messages=MSGS, **kw):
    args = {"model": "m", "prompt_version": "v1", "params": {"t": 0}} | kw
    return cache_key(messages, **args)


# --- the key (pure): anything that can change the reply must change the key --------------------


def test_the_same_inputs_make_the_same_key():
    assert key() == key([SystemMessage("rules"), HumanMessage("question + passages")])


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "other"},
        {"prompt_version": "v2"},
        {"params": {"t": 1}},
        {"messages": [SystemMessage("rules"), HumanMessage("question + other")]},
        {"messages": [SystemMessage("other"), HumanMessage("question + passages")]},
        {"messages": [HumanMessage("rules"), HumanMessage("question + passages")]},
        {"messages": [HumanMessage("question + passages"), SystemMessage("rules")]},
    ],
)
def test_changing_any_input_changes_the_key(changed):
    assert key(**changed) != key()


# --- the store -----------------------------------------------------------------------------


async def test_a_stored_reply_is_found_counted_and_priced(pool):
    cache = PgLlmCache(pool)
    assert await cache.get("k") is None

    await cache.put(
        "k", kind="extract", model="m", reply="{}", cost_usd=Decimal("0.002")
    )
    assert await cache.get("k") == "{}"
    assert await cache.get("k") == "{}"

    stats = await cache.stats()
    assert (stats.entries, stats.hits, stats.saved_usd) == (1, 2, Decimal("0.004"))


async def test_a_put_over_an_entry_replaces_the_reply_and_keeps_the_hits(pool):
    cache = PgLlmCache(pool)
    await cache.put("k", kind="extract", model="m", reply="old", cost_usd=None)
    await cache.get("k")

    await cache.put("k", kind="extract", model="m", reply="new", cost_usd=None)

    assert await cache.get("k") == "new"
    assert (await cache.stats()).hits == 2


# --- through the extractor ------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path):
    return make_settings().model_copy(update={"upload_dir": str(tmp_path / "uploads")})


@pytest.fixture
async def documents(pool, settings):
    svc = DocumentService(pool, settings)
    yield svc
    await svc.close()


@pytest.fixture
async def world(pool, settings, documents) -> World:
    w = await World(pool, settings, documents).start()
    w.service._extractor._cache = PgLlmCache(pool)  # type: ignore[union-attr]
    return w


async def fill(world: World, column, **kw):
    return await world.service.fill_cell(  # type: ignore[union-attr]
        world.workspace, world.acme.id, column.id, **kw
    )


async def test_refilling_the_same_cell_makes_no_second_model_call(world):
    await world.acme_document()
    column = await world.column()
    world.llm.reported_cost = 0.0002
    world.script(good())
    first = await fill(world, column)

    second = await fill(world, column)

    assert len(world.llm.seen) == 1
    assert first.model_calls == 1 and first.cost_usd == Decimal("0.0002")
    assert second.model_calls == 0 and second.cost_usd == Decimal(0)
    assert second.cell.status == "answered" and second.cell.value == first.cell.value
    quotes = [e.quote for e in second.cell.evidence]
    assert quotes == [e.quote for e in first.cell.evidence]


async def test_a_cache_hit_writes_no_trace_row(world, pool):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await fill(world, column)
    sql = "select count(*) from traces where kind = 'extract'"
    before = await pool.fetchval(sql)

    await fill(world, column)

    assert await pool.fetchval(sql) == before


async def test_refresh_asks_the_model_again_and_replaces_the_entry(world):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await fill(world, column)

    await fill(world, column, refresh=True)
    await fill(world, column)

    assert (
        len(world.llm.seen) == 2
    )  # the refresh asked; the third fill hit the new entry


async def test_an_edited_column_misses_the_cache(world):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await fill(world, column)
    await world.service.update_column(  # type: ignore[union-attr]
        world.workspace,
        column.id,
        ColumnPatch(prompt="How many employees work at the company?"),
    )

    await fill(world, column)

    assert len(world.llm.seen) == 2


async def test_different_document_text_misses_the_cache(world):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await fill(world, column)
    await world.acme_document(
        "Acme also reports approximately 166,000 full-time equivalent employees; "
        "employees work at the company daily."
    )

    await fill(world, column)

    assert len(world.llm.seen) == 2  # new passages in the prompt: a different key


async def test_a_broken_reply_is_not_cached(world, pool):
    await world.acme_document()
    column = await world.column()
    world.script("this is not json", good())

    await fill(world, column)

    assert (
        await pool.fetchval("select count(*) from llm_cache") == 1
    )  # only the good retry


async def test_a_cached_reply_is_still_checked_against_the_documents(world, pool):
    await world.acme_document()
    column = await world.column()
    world.script(good())
    await fill(world, column)
    await pool.execute(
        "update llm_cache set reply = $1", good(quote="employs ten thousand robots")
    )

    cell = (await fill(world, column)).cell

    # the poisoned quote failed verification; the retry (a different prompt, so a miss) went to the
    # model, and no evidence with the invented quote was kept
    assert len(world.llm.seen) == 2
    assert all("robots" not in e.quote for e in cell.evidence)


async def test_a_failing_cache_never_breaks_a_fill(world):
    await world.acme_document()
    column = await world.column()
    world.script(good())

    class Broken:
        async def get(self, key):
            raise RuntimeError("db down")

        async def put(self, *a, **k):
            raise RuntimeError("db down")

    world.service._extractor._cache = Broken()  # type: ignore[union-attr]

    assert (await fill(world, column)).cell.status == "answered"
