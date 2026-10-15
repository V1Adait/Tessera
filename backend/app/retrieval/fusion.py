"""Reciprocal Rank Fusion: merge several ranked lists into one, using only the ranks.

    score(item) = sum over the lists that contain it of   weight / (k + rank)      (rank starts at 1)

Why ranks and not scores: a full-text rank and a vector similarity are on different scales and mean
different things, so adding or averaging them is meaningless. A rank is comparable everywhere. An
item that appears high in BOTH lists beats one that is first in only one, which is the point of
running two searches that fail in different ways.

`k` softens the top: with k = 60 (the usual value) rank 1 is worth 1/61 and rank 2 is 1/62, so one
list cannot dominate by a single lucky first place. A smaller k makes the top ranks count more.
"""

from collections.abc import Hashable, Sequence


def rrf[K: Hashable](
    rankings: Sequence[Sequence[K]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """(item, fused score), best first. Items are compared by equality.

    Ties are broken by the best rank the item had in any list, then by which list saw it first,
    so the result is deterministic. An item repeated inside one list counts once, at its first
    (best) position. Raises ValueError on a bad `k` or a `weights` list of the wrong length."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")

    scores: dict[K, float] = {}
    best_rank: dict[K, int] = {}
    first_seen: dict[K, int] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        seen: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            first_seen.setdefault(item, len(first_seen))

    ordered = sorted(scores, key=lambda i: (-scores[i], best_rank[i], first_seen[i]))
    return [(item, scores[item]) for item in ordered]
