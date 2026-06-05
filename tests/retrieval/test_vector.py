"""Tests for vector retrieval with mocked Chroma and Azure OpenAI."""
from __future__ import annotations

from unittest.mock import MagicMock

from litagent.retrieval.vector import vector_search


def _fake_embedding(dim: int = 8) -> list[float]:
    return [0.1] * dim


def _make_chroma_mock(
    ids: list[str],
    docs: list[str],
    distances: list[float],
    metadatas: list[dict[str, str]],
) -> MagicMock:
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [ids],
        "documents": [docs],
        "distances": [distances],
        "metadatas": [metadatas],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


def _make_openai_mock(embedding: list[float] | None = None) -> MagicMock:
    item = MagicMock()
    item.embedding = embedding or _fake_embedding()
    response = MagicMock()
    response.data = [item]
    client = MagicMock()
    client.embeddings.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_retrieval_results() -> None:
    chroma = _make_chroma_mock(
        ids=["W1__chunk0", "W2__chunk0"],
        docs=["RAG text one.", "RAG text two."],
        distances=[0.1, 0.3],
        metadatas=[{"paper_id": "W1", "chunk_index": "0"}, {"paper_id": "W2", "chunk_index": "0"}],
    )
    openai = _make_openai_mock()
    results = vector_search("RAG hallucination", chroma, openai, "papers", "embed-model", top_k=5)
    assert len(results) == 2
    assert results[0].chunk_id == "W1__chunk0"
    assert results[0].paper_id == "W1"
    assert results[0].sources == ["vector"]


def test_score_is_one_minus_distance() -> None:
    chroma = _make_chroma_mock(
        ids=["W1__chunk0"],
        docs=["text"],
        distances=[0.2],
        metadatas=[{"paper_id": "W1", "chunk_index": "0"}],
    )
    openai = _make_openai_mock()
    results = vector_search("query", chroma, openai, "papers", "embed-model", top_k=1)
    assert abs(results[0].score - 0.8) < 1e-6


def test_large_distance_clamped_to_zero() -> None:
    chroma = _make_chroma_mock(
        ids=["W1__chunk0"],
        docs=["text"],
        distances=[1.5],
        metadatas=[{"paper_id": "W1", "chunk_index": "0"}],
    )
    openai = _make_openai_mock()
    results = vector_search("query", chroma, openai, "papers", "embed-model", top_k=1)
    assert results[0].score == 0.0


def test_embedding_call_uses_correct_model() -> None:
    chroma = _make_chroma_mock([], [], [], [])
    openai = _make_openai_mock()
    vector_search("my query", chroma, openai, "papers", "my-embed-model", top_k=5)
    openai.embeddings.create.assert_called_once_with(
        input=["my query"], model="my-embed-model"
    )


def test_empty_chroma_response_returns_empty_list() -> None:
    chroma = _make_chroma_mock([], [], [], [])
    openai = _make_openai_mock()
    results = vector_search("query", chroma, openai, "papers", "model", top_k=5)
    assert results == []
