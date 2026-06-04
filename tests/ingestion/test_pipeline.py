"""Integration tests for the ingestion pipeline with mocked external dependencies."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from litagent.config import Config, load_config
from litagent.ingestion.pipeline import (
    IngestionStats,
    _load_papers_jsonl,
    _save_papers_jsonl,
    run_ingestion,
)
from litagent.ingestion.schema import Paper


def _make_papers(n: int = 3) -> list[Paper]:
    return [
        Paper(
            id=f"W{i}",
            title=f"Paper {i} about RAG",
            abstract="Retrieval augmented generation improves factual accuracy. " * 5,
            year=2023,
            citations=[f"W{i+1}"] if i < n - 1 else [],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def test_save_and_load_papers_roundtrip(tmp_path: Path) -> None:
    papers = _make_papers(5)
    path = str(tmp_path / "papers.jsonl")
    _save_papers_jsonl(papers, path)
    loaded = _load_papers_jsonl(path)
    assert len(loaded) == 5
    assert loaded[0].id == papers[0].id
    assert loaded[0].abstract == papers[0].abstract


# ---------------------------------------------------------------------------
# Full pipeline (external deps mocked)
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_config(tmp_path: Path) -> Config:
    cfg = load_config()
    # Override paths to tmp so test doesn't write to repo
    cfg.ingestion.raw_output_path = str(tmp_path / "papers.jsonl")
    cfg.chroma.persist_directory = str(tmp_path / "chroma")
    return cfg


def test_run_ingestion_returns_stats(minimal_config: Config, tmp_path: Path) -> None:
    papers = _make_papers(3)
    fake_embedding = [0.1] * 1536

    # Mock Azure OpenAI — return exactly as many items as texts were passed in
    fake_item = MagicMock()
    fake_item.embedding = fake_embedding

    def _fake_embed(**kwargs: object) -> MagicMock:
        n = len(kwargs.get("input", []))  # type: ignore[arg-type]
        return MagicMock(data=[fake_item] * n)

    mock_openai = MagicMock()
    mock_openai.embeddings.create.side_effect = _fake_embed

    # Mock Chroma collection
    mock_collection = MagicMock()
    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_collection

    # Mock Neo4j driver + session
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.__enter__ = MagicMock(return_value=mock_driver)
    mock_driver.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    with (
        patch("litagent.ingestion.pipeline.fetch_papers", return_value=papers),
        patch("litagent.ingestion.pipeline.AzureOpenAI", return_value=mock_openai),
        patch("litagent.ingestion.pipeline.chromadb.PersistentClient", return_value=mock_chroma),
        patch("litagent.ingestion.pipeline.GraphDatabase.driver", return_value=mock_driver),
        patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "fake-key",
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_PASSWORD": "fake",
            },
        ),
    ):
        stats = run_ingestion(minimal_config)

    assert isinstance(stats, IngestionStats)
    assert stats.papers_fetched == 3
    assert stats.papers_with_abstract == 3
    assert stats.chunks_created > 0
    assert stats.chunks_embedded == stats.chunks_created
    assert stats.graph_nodes == 3
    # CITES edges: W0→W1 and W1→W2 (W2 has no citations) = 2 edges
    assert stats.graph_edges == 2


def test_checkpoint_is_written(minimal_config: Config) -> None:
    papers = _make_papers(2)
    fake_embedding = [0.0] * 1536

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=fake_embedding)] * 10
    )
    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection.return_value = MagicMock()
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.__enter__ = MagicMock(return_value=mock_driver)
    mock_driver.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    with (
        patch("litagent.ingestion.pipeline.fetch_papers", return_value=papers),
        patch("litagent.ingestion.pipeline.AzureOpenAI", return_value=mock_openai),
        patch("litagent.ingestion.pipeline.chromadb.PersistentClient", return_value=mock_chroma),
        patch("litagent.ingestion.pipeline.GraphDatabase.driver", return_value=mock_driver),
        patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "fake-key",
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_PASSWORD": "fake",
            },
        ),
    ):
        run_ingestion(minimal_config)

    assert Path(minimal_config.ingestion.raw_output_path).exists()
