"""Which parents get read after a search: grouped, ranked by best hit, capped at K."""

from uuid import UUID, uuid4

from backend.app.chunking.expand import rank_parents
from backend.app.db.records import ChunkHit, ChunkRecord


def hit(section_id: UUID, score: float, ordinal: int = 0) -> ChunkHit:
    chunk = ChunkRecord(
        id=uuid4(),
        document_id=uuid4(),
        section_id=section_id,
        ordinal=ordinal,
        kind="prose",
        text="t",
        search_text="s\nt",
        section_path=[],
        block_seqs=[0],
        char_start=0,
        char_end=1,
        token_count=1,
        page_start=None,
        page_end=None,
        sheet=None,
        slide=None,
        cell_range=None,
        units=None,
        confidence=1.0,
        flags=[],
        content_hash="h",
    )
    return ChunkHit(chunk=chunk, filename="f.pdf", score=score)


def test_no_hits_means_no_parents():
    assert rank_parents([], 5) == []


def test_hits_in_one_parent_count_once():
    a = uuid4()

    ranked = rank_parents([hit(a, 0.9), hit(a, 0.5), hit(a, 0.7)], 5)

    assert len(ranked) == 1
    assert ranked[0].best_score == 0.9
    assert [h.score for h in ranked[0].hits] == [0.9, 0.7, 0.5]  # best first


def test_parents_are_ranked_by_their_best_hit():
    a, b, c = uuid4(), uuid4(), uuid4()

    ranked = rank_parents([hit(a, 0.2), hit(b, 0.9), hit(c, 0.5), hit(a, 0.6)], 5)

    assert [r.section_id for r in ranked] == [b, a, c]


def test_only_the_top_k_parents_are_kept():
    sections = [uuid4() for _ in range(8)]
    hits = [hit(s, 1.0 - i / 10) for i, s in enumerate(sections)]

    ranked = rank_parents(hits, 5)

    assert [r.section_id for r in ranked] == sections[:5]


def test_k_counts_parents_not_hits():
    a, b = uuid4(), uuid4()
    hits = [hit(a, 0.9 - i / 100) for i in range(6)] + [hit(b, 0.1)]

    ranked = rank_parents(hits, 2)

    assert [r.section_id for r in ranked] == [
        a,
        b,
    ]  # six hits in a, and b still gets in


def test_ties_keep_the_order_the_hits_arrived_in():
    a, b = uuid4(), uuid4()

    assert [r.section_id for r in rank_parents([hit(a, 0.5), hit(b, 0.5)], 5)] == [a, b]
    assert [r.section_id for r in rank_parents([hit(b, 0.5), hit(a, 0.5)], 5)] == [b, a]


def test_a_zero_or_negative_k_gives_nothing():
    assert rank_parents([hit(uuid4(), 1.0)], 0) == []
    assert rank_parents([hit(uuid4(), 1.0)], -1) == []
