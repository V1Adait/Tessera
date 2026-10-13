"""Decide the question's type, and say how we decided.

    try the classifiers in order (the primary first)   -> a Prediction
        all failed          -> SAFE_TYPE, flagged
    confidence >= threshold -> trust the label
    confidence <  threshold -> ask a second classifier
        it agrees           -> trust the label
        it disagrees/fails  -> SAFE_TYPE (the most general path), flagged

The router never raises: a question must always get SOME route, and the Routing records why.
"""

import logging

from backend.app.query.classes import SAFE_TYPE
from backend.app.query.classify import Classifier
from backend.app.query.llm_json import ModelCallError
from backend.app.query.models import Prediction, Routing

logger = logging.getLogger(__name__)


async def _first_answer(
    classifiers: list[Classifier], question: str
) -> tuple[Prediction | None, int]:
    """The first classifier that answers, and its position in the list."""
    for position, classifier in enumerate(classifiers):
        try:
            return await classifier.classify(question), position
        except ModelCallError as exc:
            logger.warning("classifier %s failed: %s", classifier.name, exc.code)
    return None, len(classifiers)


async def route(
    question: str,
    classifiers: list[Classifier],
    *,
    threshold: float,
    second_opinion: Classifier | None = None,
) -> Routing:
    prediction, position = await _first_answer(classifiers, question)
    if prediction is None:
        return Routing(
            question_type=SAFE_TYPE,
            prediction=None,
            fell_back=True,
            reason="every classifier failed; using the general path",
        )

    fell_back = position > 0
    if prediction.confidence >= threshold:
        return Routing(
            question_type=prediction.label,
            prediction=prediction,
            fell_back=fell_back,
            reason=f"{prediction.classifier} said {prediction.label} "
            f"at {prediction.confidence:.2f}",
        )

    opinion = await _second_opinion(question, prediction, second_opinion)
    if opinion is not None and opinion.label == prediction.label:
        return Routing(
            question_type=prediction.label,
            prediction=prediction,
            second_opinion=opinion,
            low_confidence=True,
            fell_back=fell_back,
            reason=f"low confidence ({prediction.confidence:.2f}) but "
            f"{opinion.classifier} agreed: {prediction.label}",
        )
    return Routing(
        question_type=SAFE_TYPE,
        prediction=prediction,
        second_opinion=opinion,
        low_confidence=True,
        fell_back=fell_back,
        reason=f"low confidence ({prediction.confidence:.2f}) for {prediction.label} "
        "and no agreement; using the general path",
    )


async def _second_opinion(
    question: str, first: Prediction, classifier: Classifier | None
) -> Prediction | None:
    if classifier is None or classifier.name == first.classifier:
        return None  # asking the same model again would add nothing
    try:
        return await classifier.classify(question)
    except ModelCallError as exc:
        logger.warning("second opinion %s failed: %s", classifier.name, exc.code)
        return None
