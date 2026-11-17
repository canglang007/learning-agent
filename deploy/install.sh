#!/bin/sh
# 服务器端一键安装：装依赖、起 systemd 服务、做自检。
#
# 用法（在解压出来的目录里，用 root）：
#   ./deploy/install.sh
#
# 前置：/srv/learning-agent-trial/.env 已经填好 LA_API_KEY，以及 LA_AUTH=accounts
# （推荐）或 LA_ACCESS_TOKEN（共享口令）。
# 这个脚本不会覆盖已有的 .env，也不会把密钥打到屏幕上。
#
# 内存小于 1.8 G 的机器（Azure 免费档 B1s/B2ats_v2 是 1 G）会自动补一块
# 2 G swap，否则 uv 解包依赖和 ffmpeg 转码都可能被 OOM 杀掉。
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

if [ "$(id -u)" != "0" ]; then
    echo "需要 root：sudo $0" >&2
    exit 1
fi

echo "== 1/6 内存与 swap =="
mem_mb=$(awk '/^MemTotal:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
swap_mb=$(awk '/^SwapTotal:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
echo "内存 ${mem_mb} MB，现有 swap ${swap_mb} MB"
if [ "$mem_mb" -gt 0 ] && [ "$mem_mb" -lt 1800 ] && [ "$swap_mb" -lt 1024 ]; then
    echo "内存偏小，补 2 G swap"
    if [ ! -f /swapfile ]; then
        if ! fallocate -l 2G /swapfile 2>/dev/null; then
            dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
        fi
        chmod 600 /swapfile
        mkswap /swapfile >/dev/null
    fi
    swapon /swapfile 2>/dev/null || true
    grep -q '^/swapfile' /etc/fstab 2>/dev/null || echo '/swapfile none swap sw 0 0' >>/etc/fstab
    # 内存本来就紧，别让内核过早把常驻页换出去
    printf 'vm.swappiness=10\nvm.vfs_cache_pressure=50\n' >/etc/sysctl.d/99-learning-agent.conf
    sysctl -q -p /etc/sysctl.d/99-learning-agent.conf 2>/dev/null || true
    swap_mb=$(awk '/^SwapTotal:/ {print int($2/1024)}' /proc/meminfo)
    echo "swap 现在 ${swap_mb} MB"
else
    echo "内存够用，不用加 swap"
fi
if [ "$mem_mb" -gt 0 ] && [ "$mem_mb" -lt 1800 ]; then
    echo "提示：这台机器是试用档。大视频转码会慢，建议在 .env 里设"
    echo "      LA_MEDIA_MAX_BYTES=314572800（300 MB）和 LA_MEDIA_MAX_SECONDS=3600"
fi

echo "== 2/6 系统依赖 =="
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

echo "== 3/6 uv =="
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

echo "== 4/6 配置检查 =="
if [ ! -f .env ]; then
    echo "缺少 .env：先 cp .env.example .env 并填写 LA_API_KEY / LA_ACCESS_TOKEN" >&2
    exit 1
fi
chmod 600 .env
token=$(sed -n 's/^LA_ACCESS_TOKEN=//p' .env | head -1)
auth=$(sed -n 's/^LA_AUTH=//p' .env | head -1)
key=$(sed -n 's/^LA_API_KEY=//p' .env | head -1)
host=$(sed -n 's/^LA_HOST=//p' .env | head -1)
if [ -z "$key" ]; then
    echo "LA_API_KEY 为空，服务起来也答不了问题" >&2
    exit 1
fi
if [ -z "$token" ] && [ "$auth" != "accounts" ]; then
    echo "既没开账号模式（LA_AUTH=accounts），也没有 LA_ACCESS_TOKEN" >&2
    echo "任何拿到链接的人都能看全部会话。二选一：在 .env 写 LA_AUTH=accounts，" >&2
    echo "或者生成一个共享口令：python3 -c 'import secrets; print(secrets.token_urlsafe(9))'" >&2
    exit 1
fi
if [ "$host" != "0.0.0.0" ]; then
    echo "提示：LA_HOST=$host，只有本机能连；对外试用请改成 0.0.0.0"
fi

echo "== 5/6 依赖同步 =="
# 小内存机器上把下载和构建并发压下来，避免解包时峰值内存超限
UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 UV_LINK_MODE=copy \
    uv sync --frozen --no-dev

echo "== 6/6 服务 =="
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
echo "  GET /        -> $code（303 跳登录页是对的，200 说明没设防）"
code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 3 http://127.0.0.1:8320/api/spaces)
echo "  GET /api/spaces -> $code（401 是对的）"

if [ "$auth" = "accounts" ] && [ -n "$token" ]; then
    echo "  提示：LA_AUTH=accounts 时 LA_ACCESS_TOKEN 不生效，可以删掉那一行"
fi
if [ "$auth" = "accounts" ]; then
    echo
    echo "账号模式：现在还没有任何人都进不去，先发邀请码："
    echo "  uv run python scripts/accounts.py invite --note 试用 --uses 10"
    echo "  uv run python scripts/accounts.py list"
fi

echo
echo "常驻内存（RSS）："
ps -eo rss=,comm= 2>/dev/null | awk '/python/ {printf "  %d MB  %s\n", $1/1024, $2}' | head -5 || true
if command -v free >/dev/null 2>&1; then
    free -m | awk 'NR==1 || NR==2 {print "  " $0}'
fi

ip=$(curl -fsS -m 5 --noproxy '*' https://api.ipify.org 2>/dev/null || true)
echo
echo "对外地址：http://${ip:-<公网IP>}:8320"
echo "下一步：云控制台安全组放行 TCP 8320，否则外面连不进来。"
echo "日志：journalctl -u learning-agent -f"
