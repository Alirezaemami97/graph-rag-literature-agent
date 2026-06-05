"""BM25 lexical retrieval over ingested chunks."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from litagent.ingestion.schema import Chunk
from litagent.retrieval.schema import RetrievalResult


def build_bm25_index(chunks: list[Chunk]) -> tuple[BM25Okapi | None, list[Chunk]]:
    """Tokenize chunk texts and build an in-memory BM25 index.

    Returns None for the index when chunks is empty — callers must guard.
    Returns the index and the original chunks in the same order so callers
    can map ranked indices back to Chunk objects.
    """
    if not chunks:
        return None, chunks
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks


def bm25_search(
    query: str,
    index: BM25Okapi | None,
    chunks: list[Chunk],
    top_k: int,
) -> list[RetrievalResult]:
    """Return up to top_k chunks ranked by BM25 score."""
    if index is None or not chunks:
        return []
    scores = index.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        RetrievalResult(
            chunk_id=f"{chunks[i].paper_id}__chunk{chunks[i].chunk_index}",
            paper_id=chunks[i].paper_id,
            text=chunks[i].text,
            score=max(0.0, float(scores[i])),  # negative IDF (term in all docs) → clamp to 0
            sources=["bm25"],
        )
        for i in ranked
    ]
