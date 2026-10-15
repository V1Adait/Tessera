"""Question classifiers behind one interface:  await classifier.classify(question) -> Prediction.

    JevClassifier  typesafe/jev-1.13 through OpenRouter's decisions endpoint (alpha). Fast,
                   cheap, and its confidence is a real probability. The primary.
    LLMClassifier  a cheap chat model asked for JSON. Slower, and its confidence is
                   self-reported. The fallback and the second opinion.

Both raise ModelCallError when they cannot give a usable answer; the router decides what then.
Each call writes one trace row.
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from backend.app.db.records import TraceRecord
from backend.app.query.classes import CLASSES, DECOMPOSE_HINT, QuestionType
from backend.app.query.llm_json import ModelCallError, ask_json, neutralise
from backend.app.query.models import Prediction
from backend.app.query.prompts import CLASSIFY_PROMPT, CLASSIFY_PROMPT_VERSION
from backend.app.tracing import Recorder, no_recorder

logger = logging.getLogger(__name__)

OPENROUTER_ROOT = "https://openrouter.ai/api"
JEV_PROMPT_VERSION = (
    "jev-criteria-1"  # bump with CLASSES_VERSION-style changes to criteria
)
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_JEV_ATTEMPTS = 2
_JEV_BACKOFF_S = 0.3


class Classifier(Protocol):
    name: str

    async def classify(self, question: str) -> Prediction: ...


def make_jev_client(api_key: str, timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=OPENROUTER_ROOT,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )


class JevClassifier:
    """The client is passed in: production builds it once with the key, tests pass one with a
    mock transport so no network is touched."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str = "typesafe/jev-1.13",
        recorder: Recorder = no_recorder,
    ) -> None:
        self.name = model
        self._client = client
        self._model = model
        self._recorder = recorder

    def _body(self, question: str) -> dict[str, object]:
        return {
            "model": self._model,
            "state": question,
            "questions": {
                "type": {
                    "type": "choice",
                    "instructions": "What kind of question is the user asking?",
                    "criteria": CLASSES,
                },
                "needs_decomposition": {"type": "noul", "instructions": DECOMPOSE_HINT},
            },
        }

    async def _post(self, question: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(_JEV_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_JEV_BACKOFF_S * 2 ** (attempt - 1))
            try:
                response = await self._client.post(
                    "/alpha/decisions", json=self._body(question)
                )
            except httpx.TransportError as exc:  # timeouts and connection failures
                last = exc
                continue
            if response.status_code in _RETRY_STATUSES:
                last = ModelCallError("unavailable", f"HTTP {response.status_code}")
                continue
            return response
        if isinstance(last, ModelCallError):
            raise last
        raise ModelCallError("unavailable", "could not reach the classifier") from last

    async def classify(self, question: str) -> Prediction:
        started = time.perf_counter()
        tokens_in = tokens_out = None
        cost: Decimal | None = None
        code: str | None = None
        try:
            response = await self._post(question)
            if response.status_code != 200:
                code = "auth" if response.status_code in (401, 403) else "bad_request"
                logger.error(
                    "jev rejected the request: %s %s",
                    response.status_code,
                    response.text[:200],
                )
                raise ModelCallError(code, f"HTTP {response.status_code}")
            data = response.json()
            usage = data.get("usage") or {}
            tokens_in, tokens_out = (
                usage.get("input_tokens"),
                usage.get("output_tokens"),
            )
            if usage.get("cost") is not None:
                cost = Decimal(str(usage["cost"]))
            return self._parse(data)
        except ModelCallError as exc:
            code = exc.code
            raise
        except (ValueError, KeyError, TypeError) as exc:
            code = "bad_output"
            raise ModelCallError(code, "unexpected response shape") from exc
        finally:
            await self._recorder(
                TraceRecord(
                    request_id=uuid4(),
                    kind="classify",
                    status="error" if code else "ok",
                    provider="openrouter",
                    model=self._model,
                    prompt_version=JEV_PROMPT_VERSION,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost,
                    cost_source="provider"
                    if cost is not None
                    else ("unknown" if tokens_in else None),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_code=code,
                )
            )

    def _parse(self, data: dict[str, object]) -> Prediction:
        answers = data["answers"]
        assert isinstance(answers, dict)
        choice = answers["type"]
        label = choice["choice"]
        if label not in CLASSES:
            raise ModelCallError("bad_output", f"unknown class {label!r}")
        noul = answers.get("needs_decomposition", {}).get("noul")
        return Prediction(
            label=label,
            confidence=float(choice["confidence"]),
            needs_decomposition=None if noul is None else float(noul) >= 0.5,
            classifier=self.name,
        )


class _LLMReply(BaseModel):
    type: QuestionType
    confidence: float = Field(ge=0.0, le=1.0)
    needs_decomposition: bool


class LLMClassifier:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        provider: str,
        model: str,
        recorder: Recorder = no_recorder,
    ) -> None:
        self.name = model
        self._llm = llm
        self._provider = provider
        self._model = model
        self._recorder = recorder

    async def classify(self, question: str) -> Prediction:
        messages = [
            SystemMessage(content=CLASSIFY_PROMPT),
            HumanMessage(
                content=f"<question>{neutralise(question, ('question',))}</question>"
            ),
        ]
        data = await ask_json(
            self._llm,
            messages,
            recorder=self._recorder,
            kind="classify",
            provider=self._provider,
            model=self._model,
            prompt_version=CLASSIFY_PROMPT_VERSION,
        )
        try:
            reply = _LLMReply.model_validate(data)
        except ValidationError as exc:
            raise ModelCallError(
                "bad_output", "reply did not match the schema"
            ) from exc
        return Prediction(
            label=reply.type,
            confidence=reply.confidence,
            needs_decomposition=reply.needs_decomposition,
            classifier=self.name,
        )
