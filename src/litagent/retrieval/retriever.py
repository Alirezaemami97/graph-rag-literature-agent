"""HybridRetriever: orchestrates vector + BM25 + graph retrieval with RRF fusion."""
from __future__ import annotations

from typing import Any

from chromadb.api import ClientAPI as ChromaClient
from openai import AzureOpenAI

from litagent.config import Config
from litagent.ingestion.schema import Chunk
from litagent.retrieval.bm25 import bm25_search, build_bm25_index
from litagent.retrieval.fusion import reciprocal_rank_fusion
from litagent.retrieval.graph import fetch_chunks_for_papers, graph_expand
from litagent.retrieval.schema import RetrievalResponse, SearchQuery
from litagent.retrieval.vector import vector_search


class HybridRetriever:
    """Combines vector similarity, BM25 keyword, and graph traversal retrieval.

    Built once at startup (BM25 index is O(corpus)), then queried per request.
    This init/query split is how the retriever will be wired into the M3 agent tool.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        chroma_client: ChromaClient,
        embedding_client: AzureOpenAI,
        neo4j_driver: Any,
        config: Config,
    ) -> None:
        self._chunks = chunks
        self._chroma = chroma_client
        self._openai = embedding_client
        self._driver = neo4j_driver
        self._cfg = config
        self._bm25_index, _ = build_bm25_index(chunks)

    def retrieve(self, query: SearchQuery) -> RetrievalResponse:
        rc = self._cfg.retrieval

        vector_results = vector_search(
            query.text,
            self._chroma,
            self._openai,
            self._cfg.chroma.collection_name,
            self._cfg.embedding.model,
            rc.vector_top_k,
        )

        bm25_results = bm25_search(
            query.text,
            self._bm25_index,
            self._chunks,
            rc.bm25_top_k,
        )

        seed_ids = list({r.paper_id for r in vector_results})
        graph_paper_ids = graph_expand(
            seed_ids,
            self._driver,
            self._cfg.neo4j.database,
            rc.graph_hops,
            rc.graph_top_k,
        )
        graph_results = fetch_chunks_for_papers(
            graph_paper_ids,
            self._chroma,
            self._cfg.chroma.collection_name,
        )

        fused = reciprocal_rank_fusion(
            {"vector": vector_results, "bm25": bm25_results, "graph": graph_results},
            rrf_k=rc.rrf_k,
            final_top_k=rc.final_top_k,
        )

        return RetrievalResponse(
            query=query.text,
            results=fused,
            vector_candidates=len(vector_results),
            bm25_candidates=len(bm25_results),
            graph_papers_found=len(graph_paper_ids),
        )
