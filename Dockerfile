FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LA_DATA_ROOT=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# 带上 embed 扩展：镜像里没有它，向量检索与精排会静默退回关键词检索
# （.env.example 默认开两者）。模型不烘进镜像，首次启动下到 LA_DATA_ROOT/models，
# 挂载了 /app/data 就不会随容器重建重下。
RUN uv sync --frozen --no-dev --extra embed

EXPOSE 8320
CMD ["uv", "run", "--no-dev", "uvicorn", "learning_agent.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8320"]
