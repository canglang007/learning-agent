FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LA_DATA_ROOT=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8320
CMD ["uv", "run", "--no-dev", "uvicorn", "learning_agent.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8320"]
