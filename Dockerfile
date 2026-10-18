# Backend image. Dependencies are installed from uv.lock, so the image matches what is tested.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend ./backend

# Uploaded files and the small keyword-model download live here (a volume in docker-compose.yml).
RUN mkdir -p /app/data

EXPOSE 8000
# Apply migrations (safe to repeat: each runs once), then serve.
CMD ["sh", "-c", "python -m backend.app.db.migrate && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
