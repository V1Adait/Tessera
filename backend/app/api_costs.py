"""What has been spent. A thin read-only route over the traces table.

Every call that costs money writes a trace row with its cost and where the number came from
(see costs.py). This adds those rows up. Calls whose cost could not be worked out are counted on
their own line of the report and are NEVER treated as free: a total with unknown_cost_calls above
zero is a lower bound, and says so.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.app.db import repository as repo
from backend.app.db.pool import get_pool
from backend.app.llm_cache import CacheStats, PgLlmCache

router = APIRouter(tags=["costs"])

PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]


class CostReport(BaseModel):
    since: datetime
    hours: int
    total_usd: Decimal
    calls: int
    unknown_cost_calls: (
        int  # paid calls with no cost: the total is at least this much, not exactly
    )
    estimated_calls: (
        int  # calls priced from tokens x published price, not reported by the provider
    )
    lines: list[repo.CostLine]
    cache: CacheStats  # replies served from storage: no call made, nothing spent


@router.get("/costs", response_model=CostReport)
async def costs(
    pool: PoolDep, hours: Annotated[int, Query(ge=1, le=24 * 90)] = 24
) -> CostReport:
    """Spend over the last `hours`, by kind of call and model."""
    since = datetime.now(tz=UTC) - timedelta(hours=hours)
    lines = await repo.cost_summary(pool, since)
    return CostReport(
        since=since,
        hours=hours,
        total_usd=sum((line.cost_usd for line in lines), Decimal(0)),
        calls=sum(line.calls for line in lines),
        unknown_cost_calls=sum(line.unknown_cost_calls for line in lines),
        estimated_calls=sum(line.estimated_calls for line in lines),
        lines=lines,
        cache=await PgLlmCache(pool).stats(),
    )
