"""A cache for paid model calls: the same question asked the same way is answered from storage.

The key is a fingerprint of EVERYTHING that decides the reply: the model, the prompt version, the
sampling settings and the exact messages (which carry the column prompt, the entity and the retrieved
document text). Change any of them and the key changes, so nothing needs invalidating by hand: a
re-uploaded document, an edited column or a new prompt version simply misses.

What is stored is the model's raw text, not the finished cell. Verification, value parsing and the
status decision run again on every use, so a cached reply that today's code no longer accepts is
caught, not trusted.
"""

import hashlib
import json
from decimal import Decimal
from typing import Any, Protocol

import asyncpg
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


def cache_key(
    messages: list[BaseMessage],
    *,
    model: str,
    prompt_version: str,
    params: dict[str, Any] | None = None,
) -> str:
    """sha256 over a canonical JSON of the inputs. Message ROLE and text both count."""
    body = {
        "model": model,
        "prompt_version": prompt_version,
        "params": params or {},
        "messages": [[m.type, str(m.content)] for m in messages],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class LlmCache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def put(
        self, key: str, *, kind: str, model: str, reply: str, cost_usd: Decimal | None
    ) -> None: ...


class CacheStats(BaseModel):
    entries: int
    hits: int
    saved_usd: Decimal  # what the hits would have cost; entries with an unknown cost add nothing


class PgLlmCache:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, key: str) -> str | None:
        row = await self._pool.fetchrow(
            """
            UPDATE llm_cache SET hits = hits + 1, last_hit_at = now()
            WHERE key = $1 RETURNING reply
            """,
            key,
        )
        return row["reply"] if row else None

    async def put(
        self, key: str, *, kind: str, model: str, reply: str, cost_usd: Decimal | None
    ) -> None:
        # a refresh overwrites the reply but keeps the hit count
        await self._pool.execute(
            """
            INSERT INTO llm_cache (key, kind, model, reply, cost_usd)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (key) DO UPDATE SET reply = EXCLUDED.reply, cost_usd = EXCLUDED.cost_usd
            """,
            key,
            kind,
            model,
            reply,
            cost_usd,
        )

    async def stats(self) -> CacheStats:
        row = await self._pool.fetchrow(
            """
            SELECT count(*) AS entries, coalesce(sum(hits), 0) AS hits,
                   coalesce(sum(hits * cost_usd), 0) AS saved
            FROM llm_cache
            """
        )
        assert row is not None
        return CacheStats(
            entries=row["entries"], hits=int(row["hits"]), saved_usd=row["saved"]
        )
