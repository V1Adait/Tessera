import logging
from collections.abc import AsyncIterator

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_openai import ChatOpenAI

from backend.app.config import Settings
from backend.app.models import ChatRequest, Delta, Done, StreamEvent, Usage

logger = logging.getLogger(__name__)


def get_llm(settings: Settings) -> BaseChatModel:
    # The validator in Settings guarantees a provider is set; this narrows the type for mypy.
    assert settings.provider is not None
    api_key = settings.key_for(settings.provider)

    if settings.provider == "anthropic":
        return ChatAnthropic(
            model=settings.default_model,
            api_key=api_key,
            max_tokens=settings.default_max_tokens,
            timeout=settings.request_timeout,
            temperature=settings.default_temperature,
        )

    if settings.provider == "openai":
        return ChatOpenAI(
            model=settings.default_model,
            api_key=api_key,
            max_tokens=settings.default_max_tokens,
            timeout=settings.request_timeout,
            temperature=settings.default_temperature,
            stream_usage=True,
        )

    if settings.provider == "openrouter":
        return ChatOpenAI(
            model=settings.default_model,
            api_key=api_key,
            max_tokens=settings.default_max_tokens,
            timeout=settings.request_timeout,
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




async def stream_chat(request: ChatRequest, llm: BaseChatModel) -> AsyncIterator[StreamEvent]:
    """ChatRequest in, Delta events then one Done event out (our types, not LangChain's)."""
    usage: UsageMetadata | None = None

    async for chunk in llm.astream(to_lc_messages(request)):
        text = str(chunk.text)  # plain text whether content is a str or a list of blocks
        if text:  # reasoning models send many empty chunks first
            yield Delta(text=text)
        if chunk.usage_metadata is not None:  # arrives on an empty chunk near the end
            usage = chunk.usage_metadata

    if usage is None:
        # The answer already reached the user, so don't fail the call over missing bookkeeping.
        logger.warning("no usage metadata from provider; reporting zeros")
        yield Done(usage=Usage(tokens_in=0, tokens_out=0))
        return

    yield Done(usage=Usage(tokens_in=usage["input_tokens"], tokens_out=usage["output_tokens"]))

       