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
# 向量服务不可达时只提示，不改配置。曾经因为"ollama 没启动"就把 .env 换成
# 进程内小模型——那是换掉产品在用的组件，不是修环境，评测口径会跟着变。
# 环境没起来就把环境启动起来。
uv run --no-dev python - <<'PY' || true
try:
    from learning_agent.config import Settings

    s = Settings()
    if s.embedding_provider == "ollama":
        import urllib.request

        try:
            urllib.request.urlopen(s.embedding_base_url.rstrip("/") + "/api/tags", timeout=2)
        except Exception:  # noqa: BLE001 - 连不上就行，不需要具体异常类型
            print(
                f"提示：向量服务不可达（{s.embedding_base_url}），本次会退回关键词检索。"
                f"先运行 ollama serve，并确认已 `ollama pull {s.embedding_model}`；"
                "不要改成进程内小模型——那是换掉产品在用的组件，不是修环境。"
            )
except Exception:  # noqa: BLE001 - 提示脚本不能因为读配置失败就挡住启动
    pass
PY
exec uv run --no-dev python -m learning_agent.launch "$@"
