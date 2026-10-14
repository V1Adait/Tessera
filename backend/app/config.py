from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "openrouter"]
PROVIDER_ORDER: tuple[Provider, ...] = ("anthropic", "openai", "openrouter")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
    )

    provider: Provider | None = None
    default_model: str
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    max_tokens_ceiling: int
    default_max_tokens: int
    request_timeout: float
    max_retries: int = 2
    max_context_tokens: int = 100_000  # model window we allow ourselves, input + reply
    summarize_at: float = 0.7  # fraction of the input budget that triggers a summary
    recent_turns: int = 3  # turns kept word for word when a summary runs
    default_temperature: float | None = None
    # --- document uploads and parsing ---
    upload_dir: str = (
        "data/uploads"  # content-addressed storage; file names come from hashes
    )
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 1000
    max_zip_entries: int = 5000  # Word, Excel and PowerPoint files are zip archives
    max_zip_uncompressed_bytes: int = 300 * 1024 * 1024
    max_zip_ratio: int = (
        100  # a big entry may not be more than this many times its zipped size
    )
    parse_timeout_s: float = 600.0
    max_concurrent_parses: int = 2
    database_url: str | None = (
        None  # e.g. postgresql://user:password@localhost:5432/tessera
    )

    # Conversation summaries: a cheap model, called after a turn, never in the user's way.
    summary_provider: Provider = "openrouter"
    summary_model: str = "anthropic/claude-haiku-5.5"
    summary_max_tokens: int = 600  # hard cap on the summary the model may write
    summary_input_max_tokens: int = 60_000  # most history folded in by one summary call

    # Chunking: sizes in estimated tokens (3 characters each). See backend/app/chunking/.
    chunk_min_tokens: int = (
        100  # a prose chunk smaller than this is merged into a neighbour
    )
    chunk_target_tokens: int = 512  # merging small blocks stops here
    chunk_max_tokens: int = 800  # hard ceiling for one chunk
    chunk_parent_min_tokens: int = (
        300  # a parent smaller than this merges with a neighbour
    )
    chunk_parent_max_tokens: int = (
        3000  # a section bigger than this is cut into windows
    )
    # Retrieval: how many parents are read after a search. Hits in one parent count once.
    retrieval_max_parents: int = 5

    # Embeddings: vectors live in Qdrant (embedded in a local folder unless QDRANT_URL is set);
    # Postgres stays the source of truth. See backend/app/embeddings/.
    embeddings_enabled: bool = True
    embedding_model: str = (
        "openai/text-embedding-3-small"  # any id from OpenRouter's list
    )
    embedding_batch_size: int = 64  # texts per request
    embedding_timeout_s: float = 30.0
    embedding_max_chars: int = (
        24_000  # longer search text is cut for indexing (and counted)
    )
    embedding_max_chunks_per_document: int = 5000  # a guard on cost: more is refused
    embedding_max_concurrent_indexes: int = 2
    index_on_startup: bool = True  # index parsed documents that are still waiting
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None  # from the environment, never committed
    qdrant_path: str = "data/qdrant"
    qdrant_collection_prefix: str = "chunks"

    # Query handling: classify the question, then plan it. See backend/app/query/.
    router_jev_model: str = "typesafe/jev-1.13"  # pinned: an alias would move under us
    router_provider: Provider = (
        "openrouter"  # for the LLM classifier, metrics and planner
    )
    router_llm_model: str = "anthropic/claude-haiku-5.5"
    router_confidence_threshold: float = 0.8  # below this, ask for a second opinion
    router_timeout_s: float = 10.0
    router_max_tokens: int = 300
    planner_max_tokens: int = 800
    planner_max_dimensions: int = 6  # rubric dimensions the planner may pick per entity
    plan_soft_cap: int = 8  # more tasks than this need the user's confirmation
    plan_hard_cap: int = 50  # more than this is refused

    def key_for(self, provider: Provider) -> str | None:
        return getattr(self, f"{provider}_api_key")

    @model_validator(mode="after")
    def resolve_provider(self) -> Settings:
        if self.provider is not None:
            # Explicit choice: its key must exist, no silent fallback.
            if not self.key_for(self.provider):
                raise ValueError(
                    f"PROVIDER={self.provider} but {self.provider.upper()}_API_KEY is not set"
                )
            return self

        # No explicit choice: first provider wi#th a key, in PROVIDER_ORDER.
        for candidate in PROVIDER_ORDER:
            if self.key_for(candidate):
                self.provider = candidate
                return self

        names = ", ".join(f"{p.upper()}_API_KEY" for p in PROVIDER_ORDER)
        raise ValueError(f"No API key found. Set one of: {names}")


@lru_cache
def get_settings() -> Settings:
    # Required fields are filled from env/.env at runtime, which mypy can't see.
    return Settings()  # type: ignore[call-arg]
