"""Integration tests for HybridRetriever with all external deps mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

from litagent.config import load_config
from litagent.ingestion.schema import Chunk
from litagent.retrieval.retriever import HybridRetriever
from litagent.retrieval.schema import RetrievalResponse, SearchQuery


def _make_chunks(n: int = 5) -> list[Chunk]:
    return [
        Chunk(
            paper_id=f"W{i}",
            chunk_index=0,
            text=f"RAG paper {i}: retrieval augmented generation improves factual accuracy.",
        )
        for i in range(n)
    ]


def _make_openai_mock() -> MagicMock:
    item = MagicMock()
    item.embedding = [0.1] * 8
    resp = MagicMock()
    resp.data = [item]
    client = MagicMock()
    client.embeddings.create.return_value = resp
    return client


def _make_chroma_mock(chunk_ids: list[str], paper_ids: list[str], texts: list[str]) -> MagicMock:
    """Returns a Chroma client that handles both .query() and .get() calls."""
    metas = [{"paper_id": pid, "chunk_index": "0"} for pid in paper_ids]

    collection = MagicMock()
    collection.query.return_value = {
        "ids": [chunk_ids],
        "documents": [texts],
        "distances": [[0.1] * len(chunk_ids)],
        "metadatas": [metas],
    }
    collection.get.return_value = {
        "ids": chunk_ids,
        "documents": texts,
        "metadatas": metas,
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


def _make_neo4j_mock(graph_paper_ids: list[str]) -> MagicMock:
    records = []
    for pid in graph_paper_ids:
        rec = MagicMock()
        rec.__getitem__ = MagicMock(side_effect=lambda key, _pid=pid: _pid)
        records.append(rec)

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = iter(records)

    driver = MagicMock()
    driver.session.return_value = session
    return driver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_retrieval_response() -> None:
    chunks = _make_chunks(5)
    chunk_ids = [f"W{i}__chunk0" for i in range(5)]
    paper_ids = [f"W{i}" for i in range(5)]
    texts = [c.text for c in chunks]

    retriever = HybridRetriever(
        chunks=chunks,
        chroma_client=_make_chroma_mock(chunk_ids, paper_ids, texts),
        embedding_client=_make_openai_mock(),
        neo4j_driver=_make_neo4j_mock([]),
        config=load_config(),
    )
    response = retriever.retrieve(SearchQuery(text="RAG hallucination"))
    assert isinstance(response, RetrievalResponse)


def test_results_capped_at_final_top_k() -> None:
    chunks = _make_chunks(5)
    chunk_ids = [f"W{i}__chunk0" for i in range(5)]
    paper_ids = [f"W{i}" for i in range(5)]
    texts = [c.text for c in chunks]

    cfg = load_config()
    retriever = HybridRetriever(
        chunks=chunks,
        chroma_client=_make_chroma_mock(chunk_ids, paper_ids, texts),
        embedding_client=_make_openai_mock(),
        neo4j_driver=_make_neo4j_mock([]),
        config=cfg,
    )
    response = retriever.retrieve(SearchQuery(text="RAG retrieval"))
    assert len(response.results) <= cfg.retrieval.final_top_k


def test_all_result_paper_ids_are_strings() -> None:
    chunks = _make_chunks(3)
    chunk_ids = [f"W{i}__chunk0" for i in range(3)]
    paper_ids = [f"W{i}" for i in range(3)]
    texts = [c.text for c in chunks]

    retriever = HybridRetriever(
        chunks=chunks,
        chroma_client=_make_chroma_mock(chunk_ids, paper_ids, texts),
        embedding_client=_make_openai_mock(),
        neo4j_driver=_make_neo4j_mock([]),
        config=load_config(),
    )
    response = retriever.retrieve(SearchQuery(text="hallucination"))
    assert all(isinstance(r.paper_id, str) for r in response.results)


def test_stats_are_non_negative() -> None:
    chunks = _make_chunks(3)
    chunk_ids = [f"W{i}__chunk0" for i in range(3)]
    paper_ids = [f"W{i}" for i in range(3)]
    texts = [c.text for c in chunks]

    retriever = HybridRetriever(
        chunks=chunks,
        chroma_client=_make_chroma_mock(chunk_ids, paper_ids, texts),
        embedding_client=_make_openai_mock(),
        neo4j_driver=_make_neo4j_mock([]),
        config=load_config(),
    )
    response = retriever.retrieve(SearchQuery(text="RAG"))
    assert response.vector_candidates >= 0
    assert response.bm25_candidates >= 0
    assert response.graph_papers_found >= 0


def test_graph_expansion_papers_reflected_in_response() -> None:
    chunks = _make_chunks(3)
    chunk_ids = [f"W{i}__chunk0" for i in range(3)]
    paper_ids = [f"W{i}" for i in range(3)]
    texts = [c.text for c in chunks]

    retriever = HybridRetriever(
        chunks=chunks,
        chroma_client=_make_chroma_mock(chunk_ids, paper_ids, texts),
        embedding_client=_make_openai_mock(),
        neo4j_driver=_make_neo4j_mock(["W10", "W11"]),
        config=load_config(),
    )
    response = retriever.retrieve(SearchQuery(text="RAG"))
    assert response.graph_papers_found == 2


def test_sources_on_results_are_non_empty() -> None:
    chunks = _make_chunks(3)
    chunk_ids = [f"W{i}__chunk0" for i in range(3)]
    paper_ids = [f"W{i}" for i in range(3)]
    texts = [c.text for c in chunks]

    retriever = HybridRetriever(
        chunks=chunks,
        chroma_client=_make_chroma_mock(chunk_ids, paper_ids, texts),
        embedding_client=_make_openai_mock(),
        neo4j_driver=_make_neo4j_mock([]),
        config=load_config(),
    )
    response = retriever.retrieve(SearchQuery(text="RAG"))
    assert all(len(r.sources) > 0 for r in response.results)
