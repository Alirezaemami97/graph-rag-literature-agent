"""Data contracts for the hybrid retrieval layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    text: str
    seed_paper_ids: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    score: float = Field(ge=0.0)
    sources: list[str]


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    vector_candidates: int
    bm25_candidates: int
    graph_papers_found: int
