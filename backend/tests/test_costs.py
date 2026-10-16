"""Cost arithmetic and bookkeeping: pure, no database, no network."""

from decimal import Decimal

import httpx
import pytest

from backend.app.costs import (
    Cost,
    Price,
    PriceBook,
    fetch_openrouter_prices,
    reported_cost,
    resolve_cost,
)

D = Decimal


# --- what the provider reports ----------------------------------------------------------------


def test_the_reported_cost_is_read_from_the_response_metadata():
    assert reported_cost({"token_usage": {"cost": 3.8e-06}}) == D("0.0000038")


def test_a_reported_cost_of_zero_is_a_cost_not_a_missing_one():
    assert reported_cost({"token_usage": {"cost": 0}}) == D(0)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"token_usage": None},
        {"token_usage": {}},
        {"token_usage": {"cost": None}},
        {"token_usage": {"cost": -1}},
        {"token_usage": {"cost": "free"}},
        {"token_usage": {"cost": float("nan")}},
        {"token_usage": {"cost": float("inf")}},
        {"token_usage": {"cost": True}},
        {"token_usage": "not a mapping"},
    ],
)
def test_a_missing_or_nonsense_cost_is_no_cost(metadata):
    assert reported_cost(metadata) is None


def test_a_cost_sent_as_text_is_read_exactly():
    assert reported_cost({"token_usage": {"cost": "0.0000012"}}) == D("0.0000012")


# --- the price list ---------------------------------------------------------------------------

PAYLOAD = {
    "data": [
        {
            "id": "anthropic/claude-haiku-5.5",
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
        },
        {
            "id": "openai/gpt-5-mini",
            "pricing": {"prompt": "0.00000025", "completion": "0.000002"},
        },
        {
            "id": "openrouter/auto",
            "pricing": {"prompt": "-1", "completion": "-1"},
        },  # price depends on the pick
        {"id": "broken/one", "pricing": {"prompt": "x", "completion": "1"}},
        {"id": "no/pricing"},
        {"pricing": {"prompt": "1", "completion": "1"}},  # no id
        {"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}},
    ]
}


def test_the_openrouter_list_is_parsed_and_entries_without_a_usable_price_are_left_out():
    book = PriceBook.from_openrouter(PAYLOAD)

    assert len(book) == 3
    assert book.get(None, "anthropic/claude-haiku-5.5") == Price(
        D("0.0000001"), D("0.0000005")
    )
    assert book.get(None, "openrouter/auto") is None  # unknown, not free
    assert book.get(None, "free/model") == Price(
        D(0), D(0)
    )  # truly free is still a price


def test_the_cost_of_a_call_is_tokens_times_price_exactly():
    book = PriceBook.from_openrouter(PAYLOAD)

    # the call measured against the live API: 18 tokens in, 4 out, billed $0.0000038
    assert book.cost("openrouter", "anthropic/claude-haiku-5.5", 18, 4) == D(
        "0.0000038"
    )


def test_a_model_is_found_by_its_id_or_by_provider_slash_model():
    book = PriceBook({"anthropic/claude-x": Price(D(1), D(2))})

    assert book.get("anthropic", "claude-x") is not None
    assert book.get("openai", "claude-x") is None
    assert book.get("anthropic", None) is None


def test_no_price_or_no_token_count_means_no_computed_cost():
    book = PriceBook.from_openrouter(PAYLOAD)

    assert book.cost(None, "unknown/model", 10, 10) is None
    assert book.cost(None, "openai/gpt-5-mini", None, 10) is None
    assert book.cost(None, "openai/gpt-5-mini", 10, None) is None


def test_prices_written_per_million_tokens_are_converted():
    book = PriceBook.from_per_million({"claude-sonnet-5-5": (3.0, 15.0)})

    assert book.cost(None, "claude-sonnet-5-5", 1_000_000, 1_000_000) == D(18)
    assert book.cost(None, "claude-sonnet-5-5", 1000, 0) == D("0.003")


def test_your_own_prices_win_over_the_fetched_list():
    fetched = PriceBook({"m": Price(D(1), D(1))})
    mine = PriceBook({"m": Price(D(5), D(5))})

    assert fetched.merged(mine).get(None, "m") == Price(D(5), D(5))
    assert fetched.get(None, "m") == Price(
        D(1), D(1)
    )  # merging did not change the original


# --- which number is used ---------------------------------------------------------------------


def resolve(reported=None, book=None, tokens=(100, 10), model="m"):
    return resolve_cost(
        reported=reported,
        book=book,
        provider="p",
        model=model,
        tokens_in=tokens[0],
        tokens_out=tokens[1],
    )


def test_what_the_provider_reported_beats_a_computed_number():
    book = PriceBook({"m": Price(D(1), D(1))})

    assert resolve(reported=D("0.5"), book=book) == Cost(D("0.5"), "provider")


def test_a_reported_zero_is_still_the_providers_number():
    assert resolve(reported=D(0), book=PriceBook({"m": Price(D(1), D(1))})) == Cost(
        D(0), "provider"
    )


def test_without_a_reported_cost_it_is_computed_and_labelled_so():
    assert resolve(book=PriceBook({"m": Price(D("0.001"), D("0.01"))})) == Cost(
        D("0.2"), "computed"
    )


def test_with_neither_the_cost_is_unknown_not_zero():
    assert resolve(book=PriceBook()) == Cost(None, "unknown")
    assert resolve(book=None) == Cost(None, "unknown")
    assert resolve(
        book=PriceBook({"m": Price(D(1), D(1))}), tokens=(None, None)
    ) == Cost(None, "unknown")


# --- fetching the price list ------------------------------------------------------------------


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_the_price_list_is_fetched_and_parsed():
    book = await fetch_openrouter_prices(
        client=client(lambda r: httpx.Response(200, json=PAYLOAD))
    )

    assert len(book) == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=["a", "list"]),
        httpx.Response(200, json={"data": []}),
    ],
)
async def test_a_failed_or_odd_price_fetch_gives_an_empty_book_and_never_raises(
    response,
):
    book = await fetch_openrouter_prices(client=client(lambda r: response))

    assert len(book) == 0


async def test_a_network_failure_gives_an_empty_book():
    def down(request):
        raise httpx.ConnectError("no network")

    assert len(await fetch_openrouter_prices(client=client(down))) == 0
