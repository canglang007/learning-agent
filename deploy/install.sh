#!/bin/sh
# 服务器端一键安装：装依赖、起 systemd 服务、做自检。
#
# 用法（在解压出来的目录里，用 root）：
#   ./deploy/install.sh
#
# 前置：/srv/learning-agent-trial/.env 已经填好 LA_API_KEY 和 LA_ACCESS_TOKEN。
# 这个脚本不会覆盖已有的 .env，也不会把密钥打到屏幕上。
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

if [ "$(id -u)" != "0" ]; then
    echo "需要 root：sudo $0" >&2
    exit 1
fi

echo "== 1/5 系统依赖 =="
missing=""
for cmd in ffmpeg ffprobe curl; do
    command -v "$cmd" >/dev/null 2>&1 || missing="$missing $cmd"
done
if [ -n "$missing" ]; then
    echo "安装:$missing"
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y --no-install-recommends ffmpeg curl ca-certificates
    else
        echo "不是 apt 系系统，请手动安装 ffmpeg 后重跑" >&2
        exit 1
    fi
else
    echo "ffmpeg / curl 已就绪"
fi

echo "== 2/5 uv =="
if command -v uv >/dev/null 2>&1; then
    echo "uv $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    for candidate in /root/.local/bin/uv "$HOME/.local/bin/uv"; do
        if [ -x "$candidate" ]; then
            ln -sf "$candidate" /usr/local/bin/uv
            break
        fi
    done
    command -v uv >/dev/null 2>&1 || { echo "uv 安装失败" >&2; exit 1; }
    echo "uv $(uv --version)"
fi
# systemd 的 PATH 很短，统一挂一份到 /usr/local/bin
uvpath=$(command -v uv)
if [ "$uvpath" != "/usr/local/bin/uv" ]; then
    ln -sf "$uvpath" /usr/local/bin/uv
fi
echo "uv 入口：/usr/local/bin/uv -> $uvpath"

echo "== 3/5 配置检查 =="
if [ ! -f .env ]; then
    echo "缺少 .env：先 cp .env.example .env 并填写 LA_API_KEY / LA_ACCESS_TOKEN" >&2
    exit 1
fi
chmod 600 .env
token=$(sed -n 's/^LA_ACCESS_TOKEN=//p' .env | head -1)
key=$(sed -n 's/^LA_API_KEY=//p' .env | head -1)
host=$(sed -n 's/^LA_HOST=//p' .env | head -1)
if [ -z "$key" ]; then
    echo "LA_API_KEY 为空，服务起来也答不了问题" >&2
    exit 1
fi
if [ -z "$token" ]; then
    echo "LA_ACCESS_TOKEN 为空，任何拿到链接的人都能看全部会话" >&2
    echo "生成一个：python3 -c 'import secrets; print(secrets.token_urlsafe(9))'" >&2
    exit 1
fi
if [ "$host" != "0.0.0.0" ]; then
    echo "提示：LA_HOST=$host，只有本机能连；对外试用请改成 0.0.0.0"
fi

echo "== 4/5 依赖同步 =="
uv sync --frozen --no-dev

echo "== 5/5 服务 =="
cat >/etc/systemd/system/learning-agent.service <<UNIT
[Unit]
Description=Learning Agent（API + 媒体 worker）
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$root
Environment=UV_NO_PROGRESS=1
ExecStart=/usr/local/bin/uv run --no-dev python -m learning_agent.launch --host 0.0.0.0 --port 8320 --data-root ./data --no-browser
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable learning-agent >/dev/null 2>&1 || true
systemctl restart learning-agent

printf '等待启动'
i=0
while [ "$i" -lt 40 ]; do
    if curl -fsS --noproxy '*' -m 2 http://127.0.0.1:8320/healthz >/dev/null 2>&1; then
        printf '\n'
        echo "健康检查通过"
        break
    fi
    printf '.'
    i=$((i + 1))
    sleep 1
done
if [ "$i" -ge 40 ]; then
    printf '\n'
    echo "启动超时，看日志：journalctl -u learning-agent -n 80 --no-pager" >&2
    exit 1
fi

echo
echo "未登录访问应当被拦："
code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 3 http://127.0.0.1:8320/)
echo "  GET /        -> $code（303 跳登录页是对的，200 说明口令没生效）"
code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 3 http://127.0.0.1:8320/api/spaces)
echo "  GET /api/spaces -> $code（401 是对的）"

ip=$(curl -fsS -m 5 --noproxy '*' https://api.ipify.org 2>/dev/null || true)
echo
echo "对外地址：http://${ip:-<公网IP>}:8320"
echo "下一步：云控制台安全组放行 TCP 8320，否则外面连不进来。"
echo "日志：journalctl -u learning-agent -f"
