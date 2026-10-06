import logging
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from backend.app.config import Settings
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


def get_llm(settings: Settings) -> BaseChatModel:
    # The validator in Settings guarantees a provider is set; this narrows the type for mypy.
    assert settings.provider is not None
    raw_key = settings.key_for(settings.provider)
    assert raw_key  # the Settings validator guarantees the active provider has a key
    api_key = SecretStr(raw_key)

    if settings.provider == "anthropic":
        # `model` and `max_tokens` are aliases that mypy's plugin doesn't see; checked at runtime.
        return ChatAnthropic(  # type: ignore[call-arg]
            model=settings.default_model,
            api_key=api_key,
            max_tokens=settings.default_max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            temperature=settings.default_temperature,
        )

    if settings.provider == "openai":
        return ChatOpenAI(  # type: ignore[call-arg]
            model=settings.default_model,
            api_key=api_key,
            max_tokens=settings.default_max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            temperature=settings.default_temperature,
            stream_usage=True,
        )

    if settings.provider == "openrouter":
        return ChatOpenAI(  # type: ignore[call-arg]
            model=settings.default_model,
            api_key=api_key,
            max_tokens=settings.default_max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            temperature=settings.default_temperature,
            base_url="https://openrouter.ai/api/v1",
            stream_usage=True,
        )

    raise NotImplementedError(f"provider {settings.provider!r} not wired yet")


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
    request: ChatRequest, llm: BaseChatModel, request_id: UUID | None = None
) -> AsyncIterator[StreamEvent]:
    """ChatRequest in, Delta events then one Done (or one ErrorEvent) out. Our types, not LangChain's."""
    request_id = request_id or uuid4()
    usage: UsageMetadata | None = None

    try:
        async for chunk in llm.astream(to_lc_messages(request)):
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
