#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo '请先安装 uv，再运行本文件；依赖说明见 README.md。'
  exit 1
fi
# .env.example 默认开完整检索（向量 + 精排）。没装 embed 扩展时提前说清楚：
# 否则索引会进入 failed 状态，界面上看着像坏了，其实只是退回关键词检索。
if ! uv run --no-dev python -c 'import fastembed' >/dev/null 2>&1; then
  echo '提示：未安装检索扩展，本次只做关键词检索；要开向量检索与精排，先运行 uv sync --extra embed。'
fi
exec uv run --no-dev python -m learning_agent.launch "$@"
