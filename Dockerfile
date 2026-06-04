# --- Build stage: install dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install poetry==2.4.1

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-interaction --no-root

# --- Runtime stage: lean final image ---
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ src/
COPY config/ config/

ENV PYTHONPATH=/app/src

EXPOSE 8000

# M6 will add the FastAPI entrypoint; for now the pipeline is the entry point
CMD ["python", "-m", "litagent.ingestion.pipeline"]
