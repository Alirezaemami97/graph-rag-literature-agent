# alireza-literature-agent

Production-style agentic Graph RAG system for literature research. Given a research question, it returns a grounded, cited, synthesised answer backed by ~300 academic papers on Retrieval-Augmented Generation.

**Stack:** Python 3.12 · Poetry · LangGraph · Azure OpenAI · Chroma · Neo4j · LangSmith · FastAPI · Docker

## Milestones

| # | What | Status |
|---|------|--------|
| M1 | Knowledge sourcing — ingest, chunk, embed, graph | 🚧 in progress |
| M2 | Hybrid Graph RAG retrieval | pending |
| M3 | Single LangGraph agent + LangSmith tracing | pending |
| M4 | Multi-agent (Planner → Retriever → Synthesizer → Critic) | pending |
| M5 | Evaluation harness — golden set, LLM-as-judge, regression gate | pending |
| M6 | FastAPI serving + Docker + cost tracking | pending |

## Quickstart

```bash
cp .env.example .env        # fill in your Azure OpenAI + Neo4j keys
docker-compose up -d        # start Neo4j + Chroma
poetry install
poetry run python -m litagent.ingestion.pipeline   # ingest papers
poetry run pytest           # run tests
```

## Design

See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture and evaluation strategy.
