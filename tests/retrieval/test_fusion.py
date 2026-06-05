"""Tests for RRF fusion — pure math, no mocks."""
from __future__ import annotations

from litagent.retrieval.fusion import reciprocal_rank_fusion
from litagent.retrieval.schema import RetrievalResult


def _result(
    chunk_id: str, paper_id: str = "W1", score: float = 1.0, sources: list[str] | None = None
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=f"text for {chunk_id}",
        score=score,
        sources=sources or ["vector"],
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_doc_in_two_sources_outscores_doc_in_one() -> None:
    shared = _result("W1__chunk0", sources=["vector"])
    only_vector = _result("W2__chunk0", sources=["vector"])
    only_bm25 = _result("W1__chunk0", sources=["bm25"])

    results = reciprocal_rank_fusion(
        {
            "vector": [shared, only_vector],  # shared is rank 1 in vector
            "bm25": [only_bm25],              # shared is rank 1 in bm25
        },
        rrf_k=60,
        final_top_k=5,
    )
    chunk_ids = [r.chunk_id for r in results]
    # W1__chunk0 appears in both sources → higher RRF score than W2__chunk0
    assert chunk_ids[0] == "W1__chunk0"


def test_output_length_capped_at_final_top_k() -> None:
    results = reciprocal_rank_fusion(
        {"vector": [_result(f"W{i}__chunk0") for i in range(20)]},
        rrf_k=60,
        final_top_k=5,
    )
    assert len(results) <= 5


def test_deduplication_same_chunk_appears_once() -> None:
    r = _result("W1__chunk0", sources=["vector"])
    r2 = _result("W1__chunk0", sources=["bm25"])
    results = reciprocal_rank_fusion(
        {"vector": [r], "bm25": [r2]},
        rrf_k=60,
        final_top_k=10,
    )
    assert len([x for x in results if x.chunk_id == "W1__chunk0"]) == 1


def test_sources_merged_on_deduplication() -> None:
    results = reciprocal_rank_fusion(
        {
            "vector": [_result("W1__chunk0", sources=["vector"])],
            "bm25": [_result("W1__chunk0", sources=["bm25"])],
        },
        rrf_k=60,
        final_top_k=5,
    )
    assert set(results[0].sources) == {"vector", "bm25"}


def test_scores_are_positive() -> None:
    results = reciprocal_rank_fusion(
        {"vector": [_result(f"W{i}__chunk0") for i in range(5)]},
        rrf_k=60,
        final_top_k=5,
    )
    assert all(r.score > 0.0 for r in results)


def test_results_in_descending_score_order() -> None:
    results = reciprocal_rank_fusion(
        {"vector": [_result(f"W{i}__chunk0") for i in range(10)]},
        rrf_k=60,
        final_top_k=5,
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_ranked_lists_returns_empty() -> None:
    results = reciprocal_rank_fusion({}, rrf_k=60, final_top_k=5)
    assert results == []


def test_rrf_k_affects_scores() -> None:
    # Smaller k → larger score difference between ranks
    r_low_k = reciprocal_rank_fusion(
        {"vector": [_result("W1__chunk0"), _result("W2__chunk0")]},
        rrf_k=1,
        final_top_k=2,
    )
    r_high_k = reciprocal_rank_fusion(
        {"vector": [_result("W1__chunk0"), _result("W2__chunk0")]},
        rrf_k=1000,
        final_top_k=2,
    )
    gap_low = r_low_k[0].score - r_low_k[1].score
    gap_high = r_high_k[0].score - r_high_k[1].score
    assert gap_low > gap_high
