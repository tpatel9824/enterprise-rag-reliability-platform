FROM python:3.12.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_ARTIFACTS_DIR=/app/artifacts \
    LEXICAL_ARTIFACTS_DIR=/app/artifacts \
    DENSE_ARTIFACTS_DIR=/app/artifacts \
    LOCAL_IDENTITIES_PATH=/app/data/identity/local_identities.json \
    PORT=8080

WORKDIR /app

RUN groupadd --system --gid 10001 acmeflow \
    && useradd --system --uid 10001 --gid acmeflow --home-dir /app --shell /usr/sbin/nologin acmeflow

COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --requirement requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps . \
    && chown -R acmeflow:acmeflow /app

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/health', timeout=2)" || exit 1

CMD ["sh", "-c", "exec uvicorn acmeflow_rag.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
