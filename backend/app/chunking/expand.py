"""From search hits (child chunks) to the parents the model will read.

    hits (children, each with a score and a parent)
      -> group by parent               several hits in one parent count ONCE
      -> rank parents by their best hit's score
      -> keep the top K                K = settings.retrieval_max_parents (5)
      -> load each kept parent's chunks, in reading order

Why cap at K parents and not at K hits: ten hits that all sit in two sections should cost two
parents of context, not ten. And a cap bounds the prompt: at most K * parent_max_tokens.
"""

from dataclasses import dataclass, field
from uuid import UUID

from backend.app.db import repository as repo
from backend.app.db.records import ChunkHit, ChunkRecord, SectionRecord


@dataclass
class RankedParent:
    section_id: UUID
    best_score: float
    hits: list[ChunkHit] = field(
        default_factory=list
    )  # the children that matched, best first


@dataclass
class ParentView:
    """A parent ready to be read: its section, all its chunks, and which chunks matched."""

    section: SectionRecord
    chunks: list[ChunkRecord]
    hit_chunk_ids: list[UUID]
    best_score: float

    @property
    def text(self) -> str:
        """The parent's text for the prompt: its chunks in reading order, blank-line separated.
        For READING only. Quotes are checked against an individual chunk's `text`."""
        return "\n\n".join(c.text for c in self.chunks)


def rank_parents(hits: list[ChunkHit], max_parents: int) -> list[RankedParent]:
    """Group hits by parent and keep the best `max_parents`, best first.

    Ties keep the order the hits arrived in, so the result is deterministic."""
    if max_parents <= 0:
        return []
    by_parent: dict[UUID, RankedParent] = {}
    for hit in hits:
        parent = by_parent.setdefault(
            hit.chunk.section_id, RankedParent(hit.chunk.section_id, hit.score)
        )
        parent.hits.append(hit)
        parent.best_score = max(parent.best_score, hit.score)
    for parent in by_parent.values():
        parent.hits.sort(key=lambda h: -h.score)
    ranked = sorted(by_parent.values(), key=lambda p: -p.best_score)  # stable sort
    return ranked[:max_parents]


async def expand_to_parents(
    db: repo.Db, hits: list[ChunkHit], max_parents: int
) -> list[ParentView]:
    """The top parents for these hits, each with all of its chunks."""
    ranked = rank_parents(hits, max_parents)
    if not ranked:
        return []
    sections = {
        s.id: s for s in await repo.get_sections(db, [r.section_id for r in ranked])
    }
    views: list[ParentView] = []
    for parent in ranked:
        section = sections.get(parent.section_id)
        if section is None:  # deleted between the search and now: skip it, do not fail
            continue
        views.append(
            ParentView(
                section=section,
                chunks=await repo.get_section_chunks(db, parent.section_id),
                hit_chunk_ids=[h.chunk.id for h in parent.hits],
                best_score=parent.best_score,
            )
        )
    return views
