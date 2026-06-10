"""Tests for retrieval tools — mock HybridRetriever, never hit Chroma/Neo4j."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from litagent.retrieval.schema import RetrievalResponse, RetrievalResult
from litagent.tools.retrieval_tools import _format_results, make_retrieval_tools


def _make_result(paper_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper_id}__chunk0",
        paper_id=paper_id,
        text=f"Text from {paper_id}.",
        score=score,
        sources=["vector"],
    )


def _make_response(paper_ids: list[str]) -> RetrievalResponse:
    return RetrievalResponse(
        query="test query",
        results=[_make_result(pid) for pid in paper_ids],
        vector_candidates=len(paper_ids),
        bm25_candidates=0,
        graph_papers_found=0,
    )


@pytest.fixture()
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever._cfg.chroma.collection_name = "papers"
    return retriever


class TestVectorSearch:
    def test_calls_retrieve_with_query_text(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve.return_value = _make_response(["W001"])
        tools = make_retrieval_tools(mock_retriever)
        vector_search = next(t for t in tools if t.name == "vector_search")

        vector_search.invoke({"query": "hallucination in RAG"})

        call_args = mock_retriever.retrieve.call_args[0][0]
        assert call_args.text == "hallucination in RAG"
        assert call_args.seed_paper_ids == []

    def test_returns_paper_id_in_output(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve.return_value = _make_response(["W999"])
        tools = make_retrieval_tools(mock_retriever)
        vector_search = next(t for t in tools if t.name == "vector_search")

        result = vector_search.invoke({"query": "test"})

        assert "W999" in result

    def test_empty_results_returns_no_passages_message(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve.return_value = _make_response([])
        tools = make_retrieval_tools(mock_retriever)
        vector_search = next(t for t in tools if t.name == "vector_search")

        result = vector_search.invoke({"query": "obscure topic"})

        assert result == "No relevant passages found."


class TestGraphQuery:
    def test_passes_seed_ids_to_retrieve(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve.return_value = _make_response(["W002"])
        tools = make_retrieval_tools(mock_retriever)
        graph_query = next(t for t in tools if t.name == "graph_query")

        graph_query.invoke({"paper_ids": ["W001", "W003"], "query": "dense retrieval"})

        call_args = mock_retriever.retrieve.call_args[0][0]
        assert call_args.seed_paper_ids == ["W001", "W003"]
        assert call_args.text == "dense retrieval"

    def test_returns_expanded_paper_ids(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve.return_value = _make_response(["W100", "W200"])
        tools = make_retrieval_tools(mock_retriever)
        graph_query = next(t for t in tools if t.name == "graph_query")

        result = graph_query.invoke({"paper_ids": ["W001"], "query": "test"})

        assert "W100" in result
        assert "W200" in result


class TestFetchFulltext:
    def test_returns_chunk_text(self, mock_retriever: MagicMock) -> None:
        chunks = [_make_result("W500")]
        with patch(
            "litagent.tools.retrieval_tools.fetch_chunks_for_papers",
            return_value=chunks,
        ):
            tools = make_retrieval_tools(mock_retriever)
            fetch_fulltext = next(t for t in tools if t.name == "fetch_fulltext")

            result = fetch_fulltext.invoke({"paper_id": "W500"})

        assert "Text from W500." in result

    def test_missing_paper_returns_message(self, mock_retriever: MagicMock) -> None:
        with patch(
            "litagent.tools.retrieval_tools.fetch_chunks_for_papers",
            return_value=[],
        ):
            tools = make_retrieval_tools(mock_retriever)
            fetch_fulltext = next(t for t in tools if t.name == "fetch_fulltext")

            result = fetch_fulltext.invoke({"paper_id": "W999"})

        assert "No text found" in result
        assert "W999" in result


class TestFormatResults:
    def test_formats_single_result(self) -> None:
        result = _format_results([_make_result("W001", score=0.85)])
        assert "W001" in result
        assert "0.850" in result

    def test_empty_list(self) -> None:
        assert _format_results([]) == "No relevant passages found."
