"""What a paid call cost, and how we know.

Every call that costs money is traced with its cost AND the source of that number, so a total is
never mistaken for more than it is:

    provider   the provider reported it on the response (OpenRouter does, on non-streaming calls
               and for embeddings). The most trustworthy number.
    computed   tokens x the model's published price. Used when the provider did not report a cost,
               which is the case for streamed chat. An estimate: it ignores prompt-cache discounts.
    unknown    neither was available (a model with no known price). The trace says so, and
               the cost report counts these calls separately instead of treating them as free.

Costs are exact decimals, never floats: a call can cost $0.0000001 and thousands are summed.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from backend.app.db.records import CostSource

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    """USD per token."""

    prompt: Decimal
    completion: Decimal


@dataclass(frozen=True)
class Cost:
    usd: Decimal | None
    source: CostSource


def _decimal(value: Any) -> Decimal | None:
    """A non-negative finite decimal from whatever a provider sent, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    return number


def reported_cost(response_metadata: Mapping[str, Any] | None) -> Decimal | None:
    """The cost the provider put on a (non-streaming) chat response, if it did."""
    if not response_metadata:
        return None
    usage = response_metadata.get("token_usage")
    if not isinstance(usage, Mapping):
        return None
    return _decimal(usage.get("cost"))


class PriceBook:
    """Per-token prices by model id. Looked up by the model id, then by 'provider/model'."""

    def __init__(self, prices: Mapping[str, Price] | None = None) -> None:
        self._prices = dict(prices or {})

    def __len__(self) -> int:
        return len(self._prices)

    def get(self, provider: str | None, model: str | None) -> Price | None:
        if not model:
            return None
        found = self._prices.get(model)
        if found is None and provider:
            found = self._prices.get(f"{provider}/{model}")
        return found

    def cost(
        self,
        provider: str | None,
        model: str | None,
        tokens_in: int | None,
        tokens_out: int | None,
    ) -> Decimal | None:
        price = self.get(provider, model)
        if price is None or tokens_in is None or tokens_out is None:
            return None
        return price.prompt * tokens_in + price.completion * tokens_out

    @classmethod
    def from_openrouter(cls, payload: Mapping[str, Any]) -> PriceBook:
        """Parse OpenRouter's /models listing. Entries with no usable price (routers whose price
        depends on the model they pick are listed as -1) are left out, so their calls are
        'unknown' and not wrongly free."""
        prices: dict[str, Price] = {}
        for entry in payload.get("data", []):
            pricing = entry.get("pricing") or {}
            prompt, completion = (
                _decimal(pricing.get("prompt")),
                _decimal(pricing.get("completion")),
            )
            if (
                prompt is None
                or completion is None
                or not isinstance(entry.get("id"), str)
            ):
                continue
            prices[entry["id"]] = Price(prompt, completion)
        return cls(prices)

    @classmethod
    def from_per_million(cls, table: Mapping[str, tuple[float, float]]) -> PriceBook:
        """Prices written as USD per million tokens (input, output), the way providers publish
        them. For models the OpenRouter list does not cover (direct Anthropic or OpenAI keys)."""
        return cls(
            {
                model: Price(Decimal(str(i)) / _MILLION, Decimal(str(o)) / _MILLION)
                for model, (i, o) in table.items()
            }
        )

    def merged(self, other: PriceBook) -> PriceBook:
        """This book, with `other`'s prices taking precedence."""
        return PriceBook({**self._prices, **other._prices})


def resolve_cost(
    *,
    reported: Decimal | None,
    book: PriceBook | None,
    provider: str | None,
    model: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
) -> Cost:
    """The best available cost for one call, labelled with where it came from."""
    if reported is not None:
        return Cost(reported, "provider")
    computed = book.cost(provider, model, tokens_in, tokens_out) if book else None
    if computed is not None:
        return Cost(computed, "computed")
    return Cost(None, "unknown")


# --- the process-wide price list ---------------------------------------------------------------
# Set once at start-up (main.py), read by every component that traces a call. Tests set their own.

_book = PriceBook()


def set_price_book(book: PriceBook) -> None:
    global _book
    _book = book


def get_price_book() -> PriceBook:
    return _book


async def fetch_openrouter_prices(
    timeout_s: float = 10.0, client: httpx.AsyncClient | None = None
) -> PriceBook:
    """OpenRouter's public price list (no key needed). Never raises: if it cannot be fetched the
    book is empty, a warning is logged, and costs fall back to what the provider reports."""
    own = client is None
    http = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        response = await http.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        book = PriceBook.from_openrouter(response.json())
    except httpx.HTTPError, ValueError, AttributeError:
        logger.warning("could not fetch OpenRouter prices: computed costs are off")
        return PriceBook()
    finally:
        if own:
            await http.aclose()
    if len(book) == 0:
        logger.warning("OpenRouter price list was empty")
    return book
