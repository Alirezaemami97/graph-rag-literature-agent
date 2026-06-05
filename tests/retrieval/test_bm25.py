"""Tests for BM25 lexical retrieval — no mocks needed (pure Python)."""
from __future__ import annotations

from litagent.ingestion.schema import Chunk
from litagent.retrieval.bm25 import bm25_search, build_bm25_index


def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(paper_id=f"W{i}", chunk_index=0, text=text)
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# build_bm25_index
# ---------------------------------------------------------------------------

def test_index_built_from_chunks() -> None:
    chunks = _make_chunks(["RAG improves accuracy", "Graph neural networks"])
    index, returned_chunks = build_bm25_index(chunks)
    assert returned_chunks is chunks
    assert index is not None


def test_empty_corpus_returns_none_index() -> None:
    index, chunks = build_bm25_index([])
    assert index is None
    assert chunks == []


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------

def test_top_result_contains_query_keyword() -> None:
    chunks = _make_chunks([
        "RAG improves factual accuracy in language models.",
        "Graph neural networks for knowledge representation.",
        "RAG retrieval augmented generation systems.",
    ])
    index, chunks = build_bm25_index(chunks)
    results = bm25_search("RAG retrieval", index, chunks, top_k=3)
    assert len(results) > 0
    assert "RAG" in results[0].text or "retrieval" in results[0].text.lower()


def test_empty_index_returns_no_results() -> None:
    index, chunks = build_bm25_index([])
    results = bm25_search("RAG hallucination", index, chunks, top_k=5)
    assert results == []


def test_top_k_limits_results() -> None:
    chunks = _make_chunks([f"RAG paper number {i}" for i in range(10)])
    index, chunks = build_bm25_index(chunks)
    results = bm25_search("RAG", index, chunks, top_k=3)
    assert len(results) <= 3


def test_chunk_ids_formatted_correctly() -> None:
    # Need multiple docs so BM25 IDF is positive for discriminating terms
    chunks = _make_chunks([
        "RAG retrieval augmented generation technique.",
        "Graph neural networks are unrelated to this query.",
    ])
    index, chunks = build_bm25_index(chunks)
    results = bm25_search("RAG", index, chunks, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "W0__chunk0"


def test_source_is_bm25() -> None:
    chunks = _make_chunks([
        "RAG retrieval augmented generation.",
        "Unrelated document about other topics.",
    ])
    index, chunks = build_bm25_index(chunks)
    results = bm25_search("RAG", index, chunks, top_k=1)
    assert len(results) == 1
    assert results[0].sources == ["bm25"]


def test_results_descending_score() -> None:
    chunks = _make_chunks([
        "RAG RAG RAG very relevant document.",
        "RAG slightly relevant.",
        "Unrelated text about bananas.",
    ])
    index, chunks = build_bm25_index(chunks)
    results = bm25_search("RAG", index, chunks, top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
