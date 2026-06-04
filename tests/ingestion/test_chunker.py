from litagent.ingestion.chunker import chunk_paper
from litagent.ingestion.schema import Paper


def _make_paper(abstract: str | None = None) -> Paper:
    return Paper(
        id="W1",
        title="RAG: Retrieval-Augmented Generation",
        abstract=abstract,
    )


def test_chunks_produced_for_paper_with_abstract() -> None:
    paper = _make_paper("Large language models hallucinate. " * 30)
    chunks = chunk_paper(paper)
    assert len(chunks) > 0


def test_no_chunks_for_paper_without_abstract() -> None:
    paper = _make_paper(abstract=None)
    chunks = chunk_paper(paper)
    assert chunks == []


def test_chunk_indices_are_sequential() -> None:
    paper = _make_paper("This is a sentence about RAG. " * 40)
    chunks = chunk_paper(paper)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_all_chunks_reference_correct_paper_id() -> None:
    paper = _make_paper("RAG combines retrieval with generation. " * 20)
    chunks = chunk_paper(paper)
    assert all(c.paper_id == "W1" for c in chunks)


def test_chunk_size_respected() -> None:
    paper = _make_paper("word " * 500)
    chunks = chunk_paper(paper, chunk_size=512, chunk_overlap=64)
    # Each chunk should be at most chunk_size + a small buffer for the splitter
    assert all(len(c.text) <= 600 for c in chunks)


def test_title_appears_in_first_chunk() -> None:
    paper = _make_paper("Some abstract text. " * 5)
    chunks = chunk_paper(paper)
    assert len(chunks) > 0
    assert "RAG" in chunks[0].text
