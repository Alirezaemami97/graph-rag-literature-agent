"""Ingestion pipeline: fetch → checkpoint → chunk → embed → store."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI as ChromaClient
from neo4j import GraphDatabase
from openai import AzureOpenAI
from pydantic import BaseModel

from litagent.config import Config, load_config
from litagent.ingestion.chunker import chunk_paper
from litagent.ingestion.fetcher import fetch_papers
from litagent.ingestion.schema import Chunk, Paper

logger = logging.getLogger(__name__)


class IngestionStats(BaseModel):
    papers_fetched: int
    papers_with_abstract: int
    chunks_created: int
    chunks_embedded: int
    graph_nodes: int
    graph_edges: int
    failures: int


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_papers_jsonl(papers: list[Paper], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")
    logger.info("Saved %d papers to %s", len(papers), path)


def _load_papers_jsonl(path: str) -> list[Paper]:
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(Paper.model_validate_json(line))
    logger.info("Loaded %d papers from checkpoint %s", len(papers), path)
    return papers


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_chunks(
    chunks: list[Chunk], client: AzureOpenAI, model: str, batch_size: int
) -> list[list[float]]:
    """Embed all chunks in batches. Returns embeddings in same order as chunks."""
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        response = client.embeddings.create(input=texts, model=model)
        embeddings.extend([item.embedding for item in response.data])
        logger.debug("Embedded batch %d/%d", i + batch_size, len(chunks))
    return embeddings


# ---------------------------------------------------------------------------
# Chroma loader
# ---------------------------------------------------------------------------

def _load_chroma(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    chroma_client: ChromaClient,
    collection_name: str,
) -> int:
    collection = chroma_client.get_or_create_collection(collection_name)
    collection.upsert(
        ids=[f"{c.paper_id}__chunk{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,  # type: ignore[arg-type]
        metadatas=[{"paper_id": c.paper_id, "chunk_index": c.chunk_index} for c in chunks],
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Neo4j loader
# ---------------------------------------------------------------------------

def _load_neo4j(papers: list[Paper], driver: Any, database: str) -> tuple[int, int]:
    """Create Paper nodes and CITES edges. Returns (node_count, edge_count)."""
    with driver.session(database=database) as session:
        # Upsert Paper nodes
        session.run(
            """
            UNWIND $papers AS p
            MERGE (n:Paper {id: p.id})
            SET n.title = p.title,
                n.year  = p.year,
                n.venue = p.venue,
                n.doi   = p.doi
            """,
            papers=[
                {"id": pp.id, "title": pp.title, "year": pp.year, "venue": pp.venue, "doi": pp.doi}
                for pp in papers
            ],
        )

        # Upsert CITES edges (only for papers we have in corpus)
        known_ids = {pp.id for pp in papers}
        edges = [
            {"src": pp.id, "dst": cited_id}
            for pp in papers
            for cited_id in pp.citations
            if cited_id in known_ids
        ]
        if edges:
            session.run(
                """
                UNWIND $edges AS e
                MATCH (src:Paper {id: e.src})
                MATCH (dst:Paper {id: e.dst})
                MERGE (src)-[:CITES]->(dst)
                """,
                edges=edges,
            )

    return len(papers), len(edges)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ingestion(config: Config | None = None, *, from_checkpoint: bool = False) -> IngestionStats:
    if config is None:
        config = load_config()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )

    # 1. Fetch or load from checkpoint
    checkpoint = config.ingestion.raw_output_path
    if from_checkpoint and Path(checkpoint).exists():
        papers = _load_papers_jsonl(checkpoint)
    else:
        papers = fetch_papers(
            topic=config.ingestion.topic,
            max_results=config.ingestion.max_papers,
            page_size=config.ingestion.page_size,
        )
        _save_papers_jsonl(papers, checkpoint)

    # 2. Chunk
    all_chunks: list[Chunk] = []
    for paper in papers:
        all_chunks.extend(
            chunk_paper(paper, config.chunking.chunk_size, config.chunking.chunk_overlap)
        )
    papers_with_abstract = sum(1 for p in papers if p.abstract)
    logger.info("Created %d chunks from %d papers", len(all_chunks), len(papers))

    failures = 0

    # 3. Embed + load Chroma
    openai_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    embeddings = _embed_chunks(
        all_chunks, openai_client, config.embedding.model, config.embedding.batch_size
    )

    chroma_client: ChromaClient = chromadb.PersistentClient(
        path=config.chroma.persist_directory
    )
    chunks_stored = _load_chroma(
        all_chunks, embeddings, chroma_client, config.chroma.collection_name
    )
    logger.info(
        "Stored %d chunks in Chroma collection '%s'", chunks_stored, config.chroma.collection_name
    )

    # 4. Load Neo4j graph
    neo4j_uri = os.environ["NEO4J_URI"]
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ["NEO4J_PASSWORD"]

    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        node_count, edge_count = _load_neo4j(papers, driver, config.neo4j.database)
    logger.info("Graph: %d nodes, %d CITES edges", node_count, edge_count)

    return IngestionStats(
        papers_fetched=len(papers),
        papers_with_abstract=papers_with_abstract,
        chunks_created=len(all_chunks),
        chunks_embedded=len(embeddings),
        graph_nodes=node_count,
        graph_edges=edge_count,
        failures=failures,
    )


if __name__ == "__main__":
    stats = run_ingestion()
    print(stats.model_dump_json(indent=2))
