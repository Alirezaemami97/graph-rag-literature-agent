"""Vector (dense) retrieval via Chroma + Azure OpenAI embeddings."""
from __future__ import annotations

from chromadb.api import ClientAPI as ChromaClient
from openai import AzureOpenAI

from litagent.retrieval.schema import RetrievalResult


def vector_search(
    query_text: str,
    chroma_client: ChromaClient,
    embedding_client: AzureOpenAI,
    collection_name: str,
    embedding_model: str,
    top_k: int,
) -> list[RetrievalResult]:
    """Embed the query and return top_k semantically similar chunks from Chroma.

    Chroma returns L2 distances; we convert to a pseudo-similarity score
    (1 - distance) for consistency with the other retrievers. These scores
    are NOT comparable to BM25 scores — RRF fusion handles cross-source ranking.
    """
    response = embedding_client.embeddings.create(input=[query_text], model=embedding_model)
    query_embedding: list[float] = response.data[0].embedding

    collection = chroma_client.get_or_create_collection(collection_name)
    raw = collection.query(
        query_embeddings=[query_embedding],  # type: ignore[arg-type]
        n_results=top_k,
        include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
    )

    ids: list[str] = raw["ids"][0] if raw["ids"] else []
    docs: list[str] = raw["documents"][0] if raw["documents"] else []
    dists: list[float] = raw["distances"][0] if raw["distances"] else []
    _metas_raw = raw["metadatas"][0] if raw["metadatas"] else []
    metas: list[dict[str, str]] = _metas_raw  # type: ignore[assignment]

    return [
        RetrievalResult(
            chunk_id=chunk_id,
            paper_id=meta["paper_id"],
            text=doc,
            score=max(0.0, 1.0 - dist),
            sources=["vector"],
        )
        for chunk_id, doc, dist, meta in zip(ids, docs, dists, metas)
    ]
