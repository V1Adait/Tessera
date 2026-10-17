from decimal import Decimal
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
    sparse_model: str = (
        "Qdrant/bm25"  # keyword (BM25) encoder, run locally via fastembed
    )
    sparse_cache_dir: str = "data/fastembed"  # where its small one-off download is kept
    qdrant_timeout_s: float = (
        30.0  # the client default (5 s) is too short for big batches
    )
    qdrant_collection_prefix: str = "chunks"

    # Cost tracing: see costs.py. Prices come from OpenRouter's public list; add your own for models
    # it does not cover, as {"model-id": [usd_per_million_input_tokens, usd_per_million_output_tokens]}.
    fetch_model_prices: bool = True
    model_prices: dict[str, tuple[float, float]] = {}

    # Matrix cells: fill one cell from retrieved text, then check the answer in code.
    extract_provider: Provider = "openrouter"
    # Chosen on the Apple 10-K eval (20 columns, 3 runs): 15 of 17 answerable columns right, 0 wrong
    # values, 3 of 3 absent facts reported not_found, ~$0.0008 per cell. It IGNORES temperature 0
    # (see eval/probe_determinism.py), so repeat runs can differ; the checks in code and, later, the
    # result cache are what keep cells steady. gemini-2.5-flash-lite was perfectly repeatable and
    # ~3x cheaper but gave 2 wrong values before two safety-net fixes: worth re-testing.
    extract_model: str = "anthropic/claude-haiku-5.5"
    # The reply is a short JSON object, but 700 was too tight in practice: replies of exactly 700
    # tokens were cut off mid-JSON and became errors (seen live). Only tokens used are paid for.
    extract_max_tokens: int = 2000
    # When the first look finds nothing or something untrustworthy, look once more with a wider net.
    # (Found on the Apple 10-K: the answer to two columns ranked 10th and 11th, just outside the top 5.)
    extract_escalate: bool = True
    extract_escalation_parents: int = 10
    extract_escalation_candidates: int = 60
    # Identical extraction calls are answered from the llm_cache table (see llm_cache.py).
    llm_cache: bool = True
    extract_seed: int | None = (
        7  # asked of providers that honour it; see eval/probe_determinism.py
    )
    extract_candidates: int = (
        30  # chunks retrieved before they are grouped into parents
    )
    extract_context_tokens: int = (
        12_000  # most text from the documents put in one prompt
    )
    extract_low_confidence: float = (
        0.7  # evidence from a chunk at or below this is not trusted
    )

    # Matrix runs: filling many cells at once. A run shows an estimate first, has a spending cap, and
    # fills a few cells at a time so the model provider's rate limit is not hit.
    run_concurrency: int = 4
    run_default_cap_usd: Decimal = Decimal(
        "0.50"
    )  # a run stops by itself after spending this
    run_max_cap_usd: Decimal = Decimal(25)  # the highest cap a request may ask for
    run_confirm_above_usd: Decimal = Decimal(
        "0.05"
    )  # a bigger estimate needs a yes first
    run_confirm_above_cells: int = 25  # so does a bigger run, whatever it costs
    run_cell_timeout_s: float = (
        180.0  # one cell that takes longer is recorded as failed
    )
    run_calls_per_cell: float = (
        1.3  # model calls per cell: a retry or a wider look adds some
    )

    # The chat agent: the model may call tools to read and change the matrix. Every limit below is a
    # hard stop enforced in code, not a request to the model. See backend/app/agent/.
    agent_enabled: bool = True
    agent_max_steps: int = 6  # model calls in one turn (each may ask for several tools)
    agent_turn_timeout_s: float = 120.0
    agent_tool_timeout_s: float = 60.0
    agent_turn_cost_cap_usd: Decimal = Decimal("0.25")  # model spend in one turn
    session_cost_cap_usd: Decimal = Decimal(
        "2.00"
    )  # model spend in one conversation, all turns
    agent_tool_result_chars: int = 6000  # the most a tool may hand back to the model
    agent_search_passages: int = 6  # passages one search_documents call returns
    agent_action_ttl_minutes: int = (
        60  # a pending confirmation older than this is refused
    )
    agent_max_tools_per_step: int = 8  # tool calls the model may make in one step

    # Intake: after a file is parsed, one cheap model call reads its first page to say which company
    # it is about; code then finds or creates that row and attaches the file. See backend/app/intake/.
    intake_enabled: bool = True
    intake_provider: Provider = "openrouter"
    intake_model: str = "anthropic/claude-haiku-5.5"
    intake_max_tokens: int = 250
    intake_max_chars: int = 3500  # how much of the first page the model reads
    intake_min_confidence: float = (
        0.75  # below this, the file waits and a person is asked
    )

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
    # The model's reply: up to 8 columns of JSON. 1200 was too tight in practice (a reply of exactly
    # 1200 tokens was cut off mid-JSON); the prompt also asks for pack columns by key only.
    plan_columns_max_tokens: int = 3000
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
