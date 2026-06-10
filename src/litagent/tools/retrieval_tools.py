"""LangChain tools that wrap HybridRetriever for use in the LangGraph agent."""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from litagent.retrieval.graph import fetch_chunks_for_papers
from litagent.retrieval.retriever import HybridRetriever
from litagent.retrieval.schema import RetrievalResult, SearchQuery


def make_retrieval_tools(retriever: HybridRetriever) -> list[BaseTool]:
    """Build the three retrieval tools, each closed over the given retriever instance.

    Returns a list ready to be passed to llm.bind_tools() and ToolNode.
    """

    @tool
    def vector_search(query: str) -> str:
        """Search for relevant paper passages using semantic similarity and BM25 keyword matching.

        Returns the top passages with their OpenAlex paper IDs and relevance scores.
        Always call this tool first when given a new research question.
        """
        response = retriever.retrieve(SearchQuery(text=query))
        return _format_results(response.results)

    @tool
    def graph_query(paper_ids: list[str], query: str) -> str:
        """Expand the citation graph from seed paper IDs to find related papers.

        Use this after vector_search by passing the paper IDs found in those results.
        Traverses citation edges in Neo4j to surface papers not found by text search.

        paper_ids: OpenAlex paper IDs from a previous vector_search call (e.g. ['W2741809807']).
        query: the original research question, used to focus the expansion.
        """
        response = retriever.retrieve(SearchQuery(text=query, seed_paper_ids=paper_ids))
        return _format_results(response.results)

    @tool
    def fetch_fulltext(paper_id: str) -> str:
        """Retrieve all text chunks for a specific paper by its OpenAlex ID.

        Use this when you need more detail from a paper identified in earlier search results.
        paper_id: an OpenAlex paper ID (e.g. 'W2741809807').
        """
        results = fetch_chunks_for_papers(
            [paper_id],
            retriever._chroma,
            retriever._cfg.chroma.collection_name,
        )
        if not results:
            return f"No text found for paper {paper_id}."
        return "\n\n".join(r.text for r in results)

    return [vector_search, graph_query, fetch_fulltext]


def _format_results(results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant passages found."
    lines: list[str] = []
    for r in results:
        lines.append(f"[{r.paper_id}] score={r.score:.3f} sources={r.sources}")
        lines.append(r.text)
        lines.append("")
    return "\n".join(lines)
