# fakedet_system — local Haystack pipeline (no GPU, no model).
# Talks to the trained model over HTTP (FAKEDET_BACKEND_URL → HF Space).
FROM python:3.11-slim

WORKDIR /app

# Build-time deps (none beyond pip wheels). The pipeline is lightweight:
# haystack-ai, httpx, pydantic, python-dotenv, fastapi, uvicorn.
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[serve]"

# Memory SQLite persists under /app/runs (bind/volume in compose).
ENV FAKEDET_MEMORY_DB=/app/runs/fakedet_memory.sqlite

EXPOSE 9000
CMD ["uvicorn", "fakedet_system.serve.openai_api:app", \
     "--host", "0.0.0.0", "--port", "9000"]
