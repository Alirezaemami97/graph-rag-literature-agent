"""Tests for graph expansion with mocked Neo4j driver."""
from __future__ import annotations

from unittest.mock import MagicMock

from litagent.retrieval.graph import fetch_chunks_for_papers, graph_expand


def _make_neo4j_mock(paper_ids: list[str]) -> MagicMock:
    """Return a Neo4j driver mock whose session yields records with paper_id."""
    records = [MagicMock(**{"__getitem__": lambda self, key: pid}) for pid in paper_ids]
    # Make records[i]["paper_id"] work
    for rec, pid in zip(records, paper_ids):
        rec.__getitem__ = MagicMock(side_effect=lambda key, _pid=pid: _pid)

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = iter(records)

    driver = MagicMock()
    driver.session.return_value = session
    return driver


def _make_chroma_mock(
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, str]],
) -> MagicMock:
    collection = MagicMock()
    collection.get.return_value = {"ids": ids, "documents": docs, "metadatas": metas}
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


# ---------------------------------------------------------------------------
# graph_expand
# ---------------------------------------------------------------------------

def test_empty_seeds_skips_neo4j() -> None:
    driver = MagicMock()
    result = graph_expand([], driver, "neo4j", hops=2, top_k=10)
    assert result == []
    driver.session.assert_not_called()


def test_graph_expand_returns_paper_ids() -> None:
    driver = _make_neo4j_mock(["W10", "W11"])
    result = graph_expand(["W1", "W2"], driver, "neo4j", hops=2, top_k=10)
    assert result == ["W10", "W11"]


def test_graph_expand_calls_session_with_database() -> None:
    driver = _make_neo4j_mock([])
    graph_expand(["W1"], driver, database="my_db", hops=1, top_k=5)
    driver.session.assert_called_once_with(database="my_db")


def test_graph_expand_passes_hops_and_top_k() -> None:
    driver = _make_neo4j_mock([])
    graph_expand(["W1"], driver, "neo4j", hops=2, top_k=7)
    call_kwargs = driver.session.return_value.__enter__.return_value.run.call_args
    params = call_kwargs[1] if call_kwargs[1] else call_kwargs[0][1]
    assert params["hops"] == 2
    assert params["top_k"] == 7


# ---------------------------------------------------------------------------
# fetch_chunks_for_papers
# ---------------------------------------------------------------------------

def test_empty_paper_ids_skips_chroma() -> None:
    chroma = MagicMock()
    results = fetch_chunks_for_papers([], chroma, "papers")
    assert results == []
    chroma.get_or_create_collection.assert_not_called()


def test_fetch_chunks_returns_retrieval_results() -> None:
    chroma = _make_chroma_mock(
        ids=["W10__chunk0", "W10__chunk1"],
        docs=["Graph text one.", "Graph text two."],
        metas=[{"paper_id": "W10", "chunk_index": "0"}, {"paper_id": "W10", "chunk_index": "1"}],
    )
    results = fetch_chunks_for_papers(["W10"], chroma, "papers")
    assert len(results) == 2
    assert all(r.sources == ["graph"] for r in results)
    assert all(r.score == 1.0 for r in results)


def test_fetch_chunks_paper_id_set_correctly() -> None:
    chroma = _make_chroma_mock(
        ids=["W10__chunk0"],
        docs=["Some text."],
        metas=[{"paper_id": "W10", "chunk_index": "0"}],
    )
    results = fetch_chunks_for_papers(["W10"], chroma, "papers")
    assert results[0].paper_id == "W10"
    assert results[0].chunk_id == "W10__chunk0"
