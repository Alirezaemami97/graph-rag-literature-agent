import pytest
from pydantic import ValidationError

from litagent.ingestion.schema import Author, Chunk, Concept, Paper


def _valid_paper_dict() -> dict:  # type: ignore[type-arg]
    return {
        "id": "W123",
        "title": "Attention Is All You Need",
        "abstract": "We propose a novel architecture.",
        "year": 2017,
        "doi": "10.48550/arXiv.1706.03762",
        "citations": ["W456", "W789"],
        "authors": [{"id": "A1", "name": "Vaswani"}],
        "concepts": [{"id": "C1", "display_name": "Transformer", "score": 0.95}],
        "venue": "NeurIPS",
    }


def test_valid_paper_parses() -> None:
    paper = Paper.model_validate(_valid_paper_dict())
    assert paper.id == "W123"
    assert len(paper.citations) == 2
    assert paper.abstract is not None


def test_paper_missing_id_raises() -> None:
    d = _valid_paper_dict()
    del d["id"]
    with pytest.raises(ValidationError):
        Paper.model_validate(d)


def test_paper_empty_id_raises() -> None:
    d = _valid_paper_dict()
    d["id"] = "   "
    with pytest.raises(ValidationError):
        Paper.model_validate(d)


def test_paper_empty_title_raises() -> None:
    d = _valid_paper_dict()
    d["title"] = ""
    with pytest.raises(ValidationError):
        Paper.model_validate(d)


def test_paper_none_abstract_allowed() -> None:
    d = _valid_paper_dict()
    d["abstract"] = None
    paper = Paper.model_validate(d)
    assert paper.abstract is None


def test_paper_defaults_empty_lists() -> None:
    paper = Paper.model_validate({"id": "W1", "title": "A paper"})
    assert paper.citations == []
    assert paper.authors == []
    assert paper.concepts == []


def test_concept_score_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        Concept(id="C1", display_name="AI", score=1.5)


def test_chunk_empty_text_raises() -> None:
    with pytest.raises(ValidationError):
        Chunk(paper_id="W1", chunk_index=0, text="   ")


def test_valid_author_parses() -> None:
    a = Author(id="A1", name="Turing")
    assert a.name == "Turing"
