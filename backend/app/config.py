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
