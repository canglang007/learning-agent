#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo '请先安装 uv，再运行本文件；依赖说明见 README.md。'
  exit 1
fi
exec uv run --no-dev python -m learning_agent.launch "$@"
