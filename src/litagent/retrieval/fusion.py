"""Reciprocal Rank Fusion for combining results from multiple retrievers."""
from __future__ import annotations

from litagent.retrieval.schema import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievalResult]],
    rrf_k: int = 60,
    final_top_k: int = 5,
) -> list[RetrievalResult]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    score(doc) = Σ 1 / (k + rank_i) across all sources that returned it.
    A doc appearing in multiple sources scores higher than one in a single source,
    even if the latter is ranked #1 — this is the key property of RRF.
    """
    scores: dict[str, float] = {}
    sources_map: dict[str, list[str]] = {}
    results_map: dict[str, RetrievalResult] = {}

    for source_name, results in ranked_lists.items():
        for rank, result in enumerate(results, start=1):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            sources_map.setdefault(cid, []).extend(result.sources)
            results_map[cid] = result

    fused = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:final_top_k]
    return [
        RetrievalResult(
            chunk_id=cid,
            paper_id=results_map[cid].paper_id,
            text=results_map[cid].text,
            score=scores[cid],
            sources=sorted(set(sources_map[cid])),
        )
        for cid in fused
    ]
