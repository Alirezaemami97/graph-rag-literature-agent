from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    topic: str
    max_papers: int = Field(gt=0)
    page_size: int = Field(gt=0, le=200)
    raw_output_path: str


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: int = Field(gt=0)


class ChromaConfig(BaseModel):
    collection_name: str
    persist_directory: str


class Neo4jConfig(BaseModel):
    database: str


class Config(BaseModel):
    ingestion: IngestionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    neo4j: Neo4jConfig


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
