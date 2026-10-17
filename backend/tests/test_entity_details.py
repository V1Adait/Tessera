"""A company's website and ticker: normalising a domain, the update route, and its limits."""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.intake.models import Identity, normalise_domain
from backend.app.main import app
from backend.app.models import EntityUpdate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("palantir.com", "palantir.com"),
        ("https://www.Palantir.com/investors", "palantir.com"),
        ("http://palantir.com:8080/x?y=1#z", "palantir.com"),
        ("WWW.PALANTIR.COM", "palantir.com"),
        ("investors.palantir.com", "investors.palantir.com"),
        ("my-company.co.uk", "my-company.co.uk"),
        ("  palantir.com  ", "palantir.com"),
    ],
)
def test_a_website_becomes_a_bare_domain(raw, expected):
    assert normalise_domain(raw) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "palantir",
        "not a domain",
        "http://",
        "a..b",
        "-.com",
        "x" * 120 + ".com",
        None,
        5,
        ["a.com"],
        "javascript:alert(1)",
    ],
)
def test_anything_that_is_not_a_domain_is_none_never_a_guess(bad):
    assert normalise_domain(bad) is None


def test_identity_keeps_a_clean_domain_and_drops_junk():
    assert Identity(domain="https://www.palantir.com/").domain == "palantir.com"
    assert Identity(domain="palantir").domain is None


# --- the request body ---------------------------------------------------------------------------------------------


def test_an_update_cleans_the_ticker_and_the_domain():
    body = EntityUpdate(ticker=" pltr ", domain="https://www.palantir.com/x")

    assert (body.ticker, body.domain) == ("PLTR", "palantir.com")


def test_blank_or_null_means_clear_it():
    body = EntityUpdate(ticker="", domain=None)

    assert body.ticker is None and body.domain is None
    assert body.model_fields_set == {
        "ticker",
        "domain",
    }  # both were sent, so both are cleared


def test_a_field_left_out_is_not_sent():
    assert EntityUpdate(domain="palantir.com").model_fields_set == {"domain"}


@pytest.mark.parametrize(
    "bad",
    [{"ticker": "not a ticker!"}, {"ticker": "x" * 20}, {"domain": "palantir"}, {}],
)
def test_a_bad_update_is_refused(bad):
    with pytest.raises(ValidationError):
        EntityUpdate(**bad)


# --- the route ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
async def client(pool) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_pool] = lambda: pool
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def make(pool, name="Palantir", **kw):
    workspace = await repo.default_workspace_id(pool)
    return workspace, await repo.create_entity(pool, workspace, name, **kw)


async def test_the_route_sets_a_domain_and_a_ticker(client, pool):
    workspace, row = await make(pool)

    r = await client.patch(
        f"/workspaces/{workspace}/entities/{row.id}",
        json={"domain": "https://www.palantir.com", "ticker": "pltr"},
    )

    assert r.status_code == 200
    assert (r.json()["domain"], r.json()["ticker"]) == ("palantir.com", "PLTR")
    rows = await repo.list_entities(pool, workspace)
    assert [(e.domain, e.ticker) for e in rows] == [("palantir.com", "PLTR")]


async def test_leaving_a_field_out_keeps_it_and_null_clears_it(client, pool):
    workspace, row = await make(pool, ticker="PLTR", domain="palantir.com")
    url = f"/workspaces/{workspace}/entities/{row.id}"

    kept = await client.patch(url, json={"domain": "pltr.com"})
    cleared = await client.patch(url, json={"ticker": None})

    assert (kept.json()["domain"], kept.json()["ticker"]) == ("pltr.com", "PLTR")
    assert (cleared.json()["domain"], cleared.json()["ticker"]) == ("pltr.com", None)


async def test_a_ticker_another_row_has_is_refused(client, pool):
    workspace, _ = await make(pool, "Palantir", ticker="PLTR")
    other = await repo.create_entity(pool, workspace, "Acme")

    r = await client.patch(
        f"/workspaces/{workspace}/entities/{other.id}", json={"ticker": "PLTR"}
    )

    assert r.status_code == 409 and "Palantir" in r.json()["detail"]


async def test_a_row_may_keep_its_own_ticker(client, pool):
    workspace, row = await make(pool, ticker="PLTR")

    r = await client.patch(
        f"/workspaces/{workspace}/entities/{row.id}", json={"ticker": "PLTR"}
    )

    assert r.status_code == 200


async def test_the_route_404s_for_an_unknown_row_and_422s_for_junk(client, pool):
    workspace, row = await make(pool)
    base = f"/workspaces/{workspace}/entities"

    assert (
        await client.patch(f"{base}/{uuid4()}", json={"domain": "a.com"})
    ).status_code == 404
    assert (
        await client.patch(f"{base}/{row.id}", json={"domain": "nope"})
    ).status_code == 422
    assert (await client.patch(f"{base}/{row.id}", json={})).status_code == 422


async def test_a_row_of_another_workspace_is_not_touched(client, pool):
    _, row = await make(pool)
    other = await pool.fetchval(
        "insert into workspaces (name) values ('Other') returning id"
    )

    r = await client.patch(
        f"/workspaces/{other}/entities/{row.id}", json={"domain": "a.com"}
    )

    assert r.status_code == 404
    first = (await repo.list_entities(pool, await repo.default_workspace_id(pool)))[0]
    assert first.domain is None
