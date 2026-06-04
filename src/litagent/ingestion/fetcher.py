from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from litagent.ingestion.schema import Author, Concept, Paper

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openalex.org/works"


def _parse_paper(raw: dict[str, Any]) -> Paper | None:
    """Convert one OpenAlex work dict into a Paper. Returns None if essential fields missing."""
    work_id: str | None = raw.get("id")
    title: str | None = raw.get("title")
    if not work_id or not title:
        return None

    # OpenAlex stores abstract as an inverted index — reconstruct it
    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))

    authors = [
        Author(
            id=a["author"]["id"],
            name=a["author"].get("display_name") or "",
        )
        for a in raw.get("authorships", [])
        if a.get("author") and a["author"].get("id")
    ]

    concepts = [
        Concept(
            id=c["id"],
            display_name=c.get("display_name") or "",
            score=float(c.get("score", 0.0)),
        )
        for c in raw.get("concepts", [])
        if c.get("id")
    ]

    citations = [
        ref["id"]
        for ref in raw.get("referenced_works", [])
        if isinstance(ref, dict) and ref.get("id")
    ]

    venue: str | None = None
    if raw.get("primary_location") and raw["primary_location"].get("source"):
        venue = raw["primary_location"]["source"].get("display_name")

    try:
        return Paper(
            id=work_id,
            title=title,
            abstract=abstract,
            year=raw.get("publication_year"),
            doi=raw.get("doi"),
            citations=citations,
            authors=authors,
            concepts=concepts,
            venue=venue,
        )
    except Exception as exc:
        logger.warning("Skipping paper %s — validation failed: %s", work_id, exc)
        return None


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}. Reconstruct the sentence."""
    if not inverted_index:
        return None
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get_page(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def fetch_papers(topic: str, max_results: int, page_size: int = 200) -> list[Paper]:
    """Pull up to max_results papers from OpenAlex matching topic."""
    email = os.getenv("OPENALEX_EMAIL", "")
    headers = {"User-Agent": f"litagent/0.1 (mailto:{email})"} if email else {}

    papers: list[Paper] = []
    cursor = "*"

    with httpx.Client(headers=headers) as client:
        while len(papers) < max_results:
            want = min(page_size, max_results - len(papers))
            params: dict[str, Any] = {
                "filter": f"concepts.display_name:{topic}",
                "select": "id,title,abstract_inverted_index,publication_year,doi,"
                          "authorships,concepts,referenced_works,primary_location",
                "per_page": want,
                "cursor": cursor,
            }

            data = _get_page(client, params)
            results = data.get("results", [])
            if not results:
                break

            before = len(papers)
            for raw in results:
                paper = _parse_paper(raw)
                if paper is not None:
                    papers.append(paper)

            logger.info(
                "Fetched %d papers so far (got %d this page)", len(papers), len(papers) - before
            )

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

    logger.info("Fetch complete — %d valid papers for topic '%s'", len(papers), topic)
    return papers
