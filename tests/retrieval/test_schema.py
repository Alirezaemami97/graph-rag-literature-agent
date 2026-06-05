"""Tests for retrieval data contracts."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from litagent.retrieval.schema import RetrievalResponse, RetrievalResult, SearchQuery

# ---------------------------------------------------------------------------
# SearchQuery
# ---------------------------------------------------------------------------

def test_search_query_text_only() -> None:
    q = SearchQuery(text="hallucination in RAG")
    assert q.text == "hallucination in RAG"
    assert q.seed_paper_ids == []


def test_search_query_with_seeds() -> None:
    q = SearchQuery(text="RAG methods", seed_paper_ids=["W1", "W2"])
    assert len(q.seed_paper_ids) == 2


def test_search_query_requires_text() -> None:
    with pytest.raises(ValidationError):
        SearchQuery()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

def _make_result(**kwargs: object) -> RetrievalResult:
    defaults: dict[str, object] = {
        "chunk_id": "W1__chunk0",
        "paper_id": "W1",
        "text": "RAG improves factual accuracy.",
        "score": 0.85,
        "sources": ["vector"],
    }
    defaults.update(kwargs)
    return RetrievalResult.model_validate(defaults)


def test_retrieval_result_valid() -> None:
    r = _make_result()
    assert r.chunk_id == "W1__chunk0"
    assert r.score == 0.85
    assert r.sources == ["vector"]


def test_retrieval_result_score_zero_is_valid() -> None:
    r = _make_result(score=0.0)
    assert r.score == 0.0


def test_retrieval_result_negative_score_raises() -> None:
    with pytest.raises(ValidationError):
        _make_result(score=-0.1)


def test_retrieval_result_multiple_sources() -> None:
    r = _make_result(sources=["vector", "bm25", "graph"])
    assert len(r.sources) == 3


# ---------------------------------------------------------------------------
# RetrievalResponse
# ---------------------------------------------------------------------------

def test_retrieval_response_valid() -> None:
    resp = RetrievalResponse(
        query="RAG hallucination",
        results=[_make_result()],
        vector_candidates=20,
        bm25_candidates=15,
        graph_papers_found=5,
    )
    assert resp.query == "RAG hallucination"
    assert len(resp.results) == 1


def test_retrieval_response_empty_results() -> None:
    resp = RetrievalResponse(
        query="obscure topic",
        results=[],
        vector_candidates=0,
        bm25_candidates=0,
        graph_papers_found=0,
    )
    assert resp.results == []
