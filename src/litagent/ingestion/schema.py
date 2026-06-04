from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Author(BaseModel):
    id: str
    name: str


class Concept(BaseModel):
    id: str
    display_name: str
    score: float = Field(ge=0.0, le=1.0)


class Paper(BaseModel):
    id: str                          # OpenAlex work ID, e.g. "W2741809807"
    title: str
    abstract: str | None = None      # not all papers expose an abstract
    year: int | None = None
    doi: str | None = None
    citations: list[str] = Field(default_factory=list)   # OpenAlex IDs this paper cites
    authors: list[Author] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    venue: str | None = None

    @field_validator("id")
    @classmethod
    def id_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Paper id must not be empty")
        return v

    @field_validator("title")
    @classmethod
    def title_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Paper title must not be empty")
        return v


class Chunk(BaseModel):
    paper_id: str
    chunk_index: int = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must not be empty")
        return v
