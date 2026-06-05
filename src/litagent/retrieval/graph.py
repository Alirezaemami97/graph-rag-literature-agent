"""Graph retrieval: multi-hop citation expansion via Neo4j Cypher."""
from __future__ import annotations

from typing import Any

from chromadb.api import ClientAPI as ChromaClient

from litagent.retrieval.schema import RetrievalResult

# Cypher for variable-length CITES traversal.
# [:CITES*1..N] means 1 to N hops along citation edges in either direction.
# Neighbours ranked by how many seed papers they connect to — high count =
# this paper is cited by (or cites) many of our relevant papers, a strong signal.
_EXPAND_CYPHER = """
UNWIND $seed_ids AS seed_id
MATCH (p:Paper {id: seed_id})-[:CITES*1..$hops]-(neighbour:Paper)
WHERE NOT neighbour.id IN $seed_ids
RETURN neighbour.id AS paper_id, count(*) AS connections
ORDER BY connections DESC
LIMIT $top_k
"""


def graph_expand(
    seed_paper_ids: list[str],
    driver: Any,
    database: str,
    hops: int,
    top_k: int,
) -> list[str]:
    """Traverse CITES edges from seed_paper_ids and return neighbouring paper IDs.

    Returns an empty list immediately if seed_paper_ids is empty to avoid
    an unnecessary Neo4j round-trip.
    """
    if not seed_paper_ids:
        return []
    with driver.session(database=database) as session:
        result = session.run(
            _EXPAND_CYPHER,
            seed_ids=seed_paper_ids,
            hops=hops,
            top_k=top_k,
        )
        return [record["paper_id"] for record in result]


def fetch_chunks_for_papers(
    paper_ids: list[str],
    chroma_client: ChromaClient,
    collection_name: str,
) -> list[RetrievalResult]:
    """Fetch all chunks for the given paper_ids from Chroma.

    These are graph-discovered papers; they receive a neutral score of 1.0
    because RRF will re-rank them by fusion position, not raw score.
    """
    if not paper_ids:
        return []
    collection = chroma_client.get_or_create_collection(collection_name)
    raw = collection.get(
        where={"paper_id": {"$in": paper_ids}},  # type: ignore[dict-item]
        include=["documents", "metadatas"],  # type: ignore[list-item]
    )
    ids: list[str] = raw["ids"] or []
    docs: list[str] = raw["documents"] or []
    metas: list[dict[str, str]] = raw["metadatas"] or []  # type: ignore[assignment]
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            paper_id=meta["paper_id"],
            text=doc,
            score=1.0,
            sources=["graph"],
        )
        for chunk_id, doc, meta in zip(ids, docs, metas)
    ]
