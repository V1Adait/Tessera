"""The FastAPI app. Run with:  uv run uvicorn backend.app.main:app --reload"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router
from backend.app.config import get_settings
from backend.app.llm import get_default_llm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail fast: a bad .env or model setup stops the server at startup,
    # not on the first user request.
    settings = get_settings()
    get_default_llm()
    logging.getLogger(__name__).info(
        "started: provider=%s model=%s", settings.provider, settings.default_model
    )
    yield


app = FastAPI(title="Tessera", lifespan=lifespan)

# Lets the browser UI (Vite dev server) call this API from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(router)
