from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from litagent.ingestion.schema import Chunk, Paper

logger = logging.getLogger(__name__)


def chunk_paper(paper: Paper, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Split a paper's text into overlapping chunks, each prefixed with the title."""
    text = paper.abstract or ""
    if not text.strip():
        logger.debug("Paper %s has no abstract — skipping chunking", paper.id)
        return []

    # Prepend title so every chunk is self-contained when retrieved
    full_text = f"{paper.title}\n\n{text}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(full_text)

    return [
        Chunk(paper_id=paper.id, chunk_index=i, text=chunk_text)
        for i, chunk_text in enumerate(raw_chunks)
        if chunk_text.strip()
    ]
