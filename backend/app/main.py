"""The FastAPI app. Run with:  uv run uvicorn backend.app.main:app --reload"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router
from backend.app.api_workspaces import router as workspaces_router
from backend.app.config import get_settings
from backend.app.db.migrate import pending_migrations
from backend.app.db.pool import create_pool
from backend.app.llm import get_default_llm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail fast: a bad .env, model setup or database stops the server at startup,
    # not on the first user request.
    log = logging.getLogger(__name__)
    settings = get_settings()
    get_default_llm()
    log.info("started: provider=%s model=%s", settings.provider, settings.default_model)

    app.state.pool = None
    if settings.database_url:
        pool = await create_pool(settings)
        pending = await pending_migrations(pool)
        if pending:
            await pool.close()
            raise RuntimeError(
                f"database needs migrations: {', '.join(pending)}. "
                "Run: uv run python -m backend.app.db.migrate"
            )
        app.state.pool = pool
        log.info("database pool ready")
    else:
        log.warning("DATABASE_URL is not set: running without persistence")

    try:
        yield
    finally:
        if app.state.pool is not None:
            await app.state.pool.close()


app = FastAPI(title="Tessera", lifespan=lifespan)

# Lets the browser UI (Vite dev server) call this API from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Conversation-ID"],
)

app.include_router(router)
app.include_router(workspaces_router)
