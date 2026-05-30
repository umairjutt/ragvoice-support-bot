from app.models import Hit
from app.retrieval import reciprocal_rank_fusion


def make_hits(ids: list[int]) -> list[Hit]:
    return [Hit(chunk_id=i, document_id=i, text=f"chunk {i}", score=0.0) for i in ids]


def test_rrf_rewards_items_ranked_high_in_both_lists():
    dense = make_hits([1, 2, 3])
    sparse = make_hits([2, 1, 4])

    fused = reciprocal_rank_fusion(dense, sparse)

    # id 2 appears at ranks 1 and 0 -> should outrank id 1 (ranks 0 and 1 too, tie),
    # but both should beat ids that appear in only one list.
    ranked_ids = [h.chunk_id for h in fused]
    assert set(ranked_ids[:2]) == {1, 2}
    assert ranked_ids[-1] in {3, 4}


def test_rrf_deduplicates_chunks():
    dense = make_hits([1, 1, 2])
    sparse = make_hits([1])

    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [h.chunk_id for h in fused]

    assert len(ids) == len(set(ids))


def test_rrf_handles_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
