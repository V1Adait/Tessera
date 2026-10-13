"""Pull the metric phrases out of a question ('revenue', 'net debt', 'operating margin').

A cheap model reads the question; code then checks its answer. Every metric must appear in
the question text, word for word (ignoring case and spacing). A phrase the model made up, or
renamed, is dropped, so a metric in a plan can always be traced back to the user's own words.
"""

import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.query.llm_json import ModelCallError, ask_json, neutralise
from backend.app.query.prompts import METRICS_PROMPT, METRICS_PROMPT_VERSION
from backend.app.tracing import Recorder, no_recorder

logger = logging.getLogger(__name__)

MAX_METRICS = 8
MAX_METRIC_CHARS = 60


@dataclass(frozen=True)
class MetricResult:
    metrics: list[str]
    dropped: list[str]  # phrases the model returned that were not in the question
    failed: bool = (
        False  # the call itself failed (the plan still works, without metrics)
    )


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_metrics(question: str, candidates: object) -> MetricResult:
    """Keep the candidates that appear verbatim in the question. Pure code, no model."""
    if not isinstance(candidates, list):
        return MetricResult([], [])
    haystack = _squash(question)
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        phrase = _squash(item)
        if not phrase or len(phrase) > MAX_METRIC_CHARS:
            dropped.append(str(item)[:MAX_METRIC_CHARS])
        elif phrase not in haystack:
            dropped.append(item)
        elif phrase not in seen:
            seen.add(phrase)
            kept.append(phrase)
    return MetricResult(kept[:MAX_METRICS], dropped)


class MetricExtractor:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        provider: str,
        model: str,
        recorder: Recorder = no_recorder,
    ) -> None:
        self._llm = llm
        self._provider = provider
        self._model = model
        self._recorder = recorder

    async def extract(self, question: str) -> MetricResult:
        messages = [
            SystemMessage(content=METRICS_PROMPT),
            HumanMessage(
                content=f"<question>{neutralise(question, ('question',))}</question>"
            ),
        ]
        try:
            data = await ask_json(
                self._llm,
                messages,
                recorder=self._recorder,
                kind="extract",
                provider=self._provider,
                model=self._model,
                prompt_version=METRICS_PROMPT_VERSION,
            )
        except ModelCallError as exc:
            logger.warning("metric extraction failed: %s", exc.code)
            return MetricResult([], [], failed=True)
        return verify_metrics(question, data.get("metrics"))
