import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from uuid import UUID, uuid4

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from backend.app.config import Provider, Settings, get_settings
from backend.app.models import (
    ChatRequest,
    Delta,
    Done,
    ErrorCode,
    ErrorEvent,
    StreamEvent,
    Usage,
)

logger = logging.getLogger(__name__)


def get_llm(
    settings: Settings,
    *,
    provider: Provider | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> BaseChatModel:
    """Build a chat model. With no overrides it is the main chat model from the settings; the
    keyword arguments let another task (such as summarising) use a different model.

    `temperature=0` and `seed` are requests, not guarantees: some models ignore them and answer
    differently every time. Check a model with eval/probe_determinism.py before relying on it."""
    # The validator in Settings guarantees a provider is set; this narrows the type for mypy.
    chosen = provider or settings.provider
    assert chosen is not None
    raw_key = settings.key_for(chosen)
    if not raw_key:
        raise ValueError(f"no API key for provider {chosen!r}")
    api_key = SecretStr(raw_key)
    model_name = model or settings.default_model
    token_cap = max_tokens or settings.default_max_tokens
    temp = temperature if temperature is not None else settings.default_temperature

    if chosen == "anthropic":
        # `model` and `max_tokens` are aliases that mypy's plugin doesn't see; checked at runtime.
        return ChatAnthropic(  # type: ignore[call-arg]
            model=model_name,
            api_key=api_key,
            max_tokens=token_cap,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            temperature=temp,
        )

    if chosen == "openai":
        return ChatOpenAI(  # type: ignore[call-arg]
            model=model_name,
            api_key=api_key,
            max_tokens=token_cap,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            temperature=temp,
            seed=seed,
            stream_usage=True,
        )

    if chosen == "openrouter":
        return ChatOpenAI(  # type: ignore[call-arg]
            model=model_name,
            api_key=api_key,
            max_tokens=token_cap,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            temperature=temp,
            seed=seed,
            base_url="https://openrouter.ai/api/v1",
            stream_usage=True,
        )

    raise NotImplementedError(f"provider {chosen!r} not wired yet")


@lru_cache
def get_default_llm() -> BaseChatModel:
    """The shared model client, built once from the settings and reused by every request."""
    return get_llm(get_settings())


@lru_cache
def get_summary_llm() -> BaseChatModel | None:
    """The cheap model that writes conversation summaries, or None when its provider has no
    key (summaries are then simply switched off, and chat keeps working)."""
    settings = get_settings()
    if not settings.key_for(settings.summary_provider):
        logger.warning(
            "no API key for summary provider %r: conversation summaries are off",
            settings.summary_provider,
        )
        return None
    return get_llm(
        settings,
        provider=settings.summary_provider,
        model=settings.summary_model,
        max_tokens=settings.summary_max_tokens,
        temperature=0.2,  # summaries should be stable, not creative
    )


def to_lc_messages(request: ChatRequest) -> list[BaseMessage]:
    """Convert our ChatRequest into LangChain messages."""
    role_to_class: dict[str, type[BaseMessage]] = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    messages: list[BaseMessage] = []
    if request.system:
        messages.append(SystemMessage(content=request.system))
    for m in request.messages:
        messages.append(role_to_class[m.role](content=m.content))
    return messages


def classify_error(exc: Exception) -> tuple[ErrorCode, str, bool]:
    """Map a provider exception to (code, client-safe message, retryable).

    Messages are fixed strings on purpose: raw provider errors can echo request details.
    The full exception is logged server-side by the caller.
    """
    # Both SDKs use the same class names. Order matters: timeout is a kind of connection error.
    if isinstance(exc, (anthropic.AuthenticationError, openai.AuthenticationError)):
        return "auth", "The provider rejected the API key.", False
    if isinstance(exc, (anthropic.PermissionDeniedError, openai.PermissionDeniedError)):
        return (
            "auth",
            "The API key is not allowed to use this model or endpoint.",
            False,
        )
    if isinstance(exc, (anthropic.NotFoundError, openai.NotFoundError)):
        return "model_not_found", "The model was not found. Check DEFAULT_MODEL.", False
    if isinstance(exc, (anthropic.RateLimitError, openai.RateLimitError)):
        return "rate_limited", "The provider is rate limiting requests.", True
    if isinstance(exc, (anthropic.APITimeoutError, openai.APITimeoutError)):
        return "timeout", "The provider timed out.", True
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return "unavailable", "Could not reach the provider.", True
    if isinstance(exc, (anthropic.InternalServerError, openai.InternalServerError)):
        return "unavailable", "The provider had an internal error.", True
    if isinstance(exc, (anthropic.BadRequestError, openai.BadRequestError)):
        return "bad_request", "The provider rejected the request.", False
    return "unknown", "Unexpected error while calling the model.", False


async def stream_chat(
    messages: list[BaseMessage], llm: BaseChatModel, request_id: UUID | None = None
) -> AsyncIterator[StreamEvent]:
    """Finished prompt in (from build_context), Delta events then one Done or ErrorEvent out."""
    request_id = request_id or uuid4()
    usage: UsageMetadata | None = None

    try:
        async for chunk in llm.astream(messages):
            text = str(
                chunk.text
            )  # plain text whether content is a str or a list of blocks
            if text:  # reasoning models send many empty chunks first
                yield Delta(text=text)
            if (
                chunk.usage_metadata is not None
            ):  # arrives on an empty chunk near the end
                usage = chunk.usage_metadata
    except Exception as exc:
        code, message, retryable = classify_error(exc)
        logger.exception(
            "llm call failed", extra={"request_id": str(request_id), "code": code}
        )
        yield ErrorEvent(
            request_id=request_id, code=code, message=message, retryable=retryable
        )
        return

    if usage is None:
        # The answer already reached the user, so don't fail the call over missing bookkeeping.
        logger.warning(
            "no usage metadata from provider; reporting zeros",
            extra={"request_id": str(request_id)},
        )
        yield Done(request_id=request_id, usage=Usage(tokens_in=0, tokens_out=0))
        return

    yield Done(
        request_id=request_id,
        usage=Usage(tokens_in=usage["input_tokens"], tokens_out=usage["output_tokens"]),
    )
