"""The FastAPI app. Run with:  uv run uvicorn backend.app.main:app --reload"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router
from backend.app.api_documents import router as documents_router
from backend.app.api_query import router as query_router
from backend.app.api_search import router as search_router
from backend.app.api_workspaces import router as workspaces_router
from backend.app.config import Settings, get_settings
from backend.app.db import repository as repo
from backend.app.db.migrate import pending_migrations
from backend.app.db.pool import create_pool
from backend.app.documents_service import DocumentService
from backend.app.embeddings.embedder import OpenRouterEmbedder, make_embedding_client
from backend.app.embeddings.indexer import Indexer
from backend.app.embeddings.store import QdrantStore, make_qdrant_client
from backend.app.llm import get_default_llm
from backend.app.query.service import build_query_service
from backend.app.tracing import db_recorder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


def build_indexer(pool: asyncpg.Pool, settings: Settings) -> Indexer | None:
    """The vector index, or None when embeddings are off or no OpenRouter key is set."""
    if not settings.embeddings_enabled or not settings.openrouter_api_key:
        logging.getLogger(__name__).warning(
            "vector indexing is off (embeddings disabled or no OPENROUTER_API_KEY)"
        )
        return None
    client = make_embedding_client(
        settings.openrouter_api_key, settings.embedding_timeout_s
    )
    embedder = OpenRouterEmbedder(
        client, model=settings.embedding_model, recorder=db_recorder(pool)
    )
    store = QdrantStore(
        make_qdrant_client(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            path=settings.qdrant_path,
        ),
        prefix=settings.qdrant_collection_prefix,
        model=settings.embedding_model,
        payload_indexes=bool(settings.qdrant_url),
    )
    return Indexer(pool, settings, embedder, store)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail fast: a bad .env, model setup or database stops the server at startup,
    # not on the first user request.
    log = logging.getLogger(__name__)
    settings = get_settings()
    get_default_llm()
    log.info("started: provider=%s model=%s", settings.provider, settings.default_model)

    app.state.pool = None
    app.state.documents = None
    app.state.query = None
    app.state.indexer = None
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
        indexer = build_indexer(pool, settings)
        app.state.indexer = indexer
        app.state.documents = DocumentService(pool, settings, indexer=indexer)
        app.state.query = build_query_service(pool, settings)
        interrupted = await repo.recover_interrupted(pool)
        if interrupted:
            log.warning(
                "%s document(s) were mid-parse at the last shutdown; marked failed, re-parse them",
                interrupted,
            )
        log.info("database pool ready")
        if indexer is not None:
            await repo.recover_interrupted_indexing(pool)
            if settings.index_on_startup:
                waiting = await indexer.schedule_pending()
                if waiting:
                    log.info(
                        "indexing %s waiting document(s) in the background", waiting
                    )
    else:
        log.warning("DATABASE_URL is not set: running without persistence")

    try:
        yield
    finally:
        if app.state.documents is not None:
            await app.state.documents.close()
        if app.state.query is not None:
            await app.state.query.close()
        if app.state.indexer is not None:
            await app.state.indexer.close()
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
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(search_router)
