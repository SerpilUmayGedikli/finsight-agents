FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY apps/ ./apps/
COPY gateway/ ./gateway/
COPY memory/ ./memory/
RUN uv sync --frozen --no-dev

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app app \
    && mkdir -p /app/data/raw /home/app/.cache/huggingface \
    && chown -R app:app /app/data /home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]