#!/bin/sh
# 打一个试用部署包：代码 + 演示工作区，不含评测 corpus。
#
# 用法：./deploy/pack.sh [输出目录]
# 产物：dist/learning-agent-trial.tar.gz（约 50 MB）
#
# 只放对外试用需要的东西。evaluations/ 有 2.6 G，是跑评测用的，
# 传上去除了占盘和拖慢上传没有任何作用。
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$root/dist}
name=learning-agent-trial

demo=${LA_DEMO_DATA:-$root/data/demo/workspace}
if [ ! -f "$demo/workspace.db" ]; then
    echo "找不到演示工作区：$demo/workspace.db" >&2
    echo "用 LA_DEMO_DATA=/path/to/workspace 指定其他目录" >&2
    exit 1
fi

mkdir -p "$out"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
target=$stage/$name
mkdir -p "$target"

# 代码与运行所需文件
for item in src scripts pyproject.toml uv.lock README.md docs examples .env.example \
            Dockerfile docker-compose.yml .dockerignore start.sh deploy; do
    if [ -e "$root/$item" ]; then
        cp -R "$root/$item" "$target/"
    fi
done

# 演示数据放到 data/ 下，容器和直接启动都按这个路径找
mkdir -p "$target/data"
cp -R "$demo/." "$target/data/"

# 清理不需要进包的东西
find "$target" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$target" -name '*.pyc' -delete 2>/dev/null || true

tarball=$out/$name.tar.gz
tar czf "$tarball" -C "$stage" "$name"
size=$(du -h "$tarball" | cut -f1)
printf '已生成：%s（%s）\n' "$tarball" "$size"
printf '服务器上：tar xzf %s.tar.gz -C /srv && /srv/%s/deploy/install.sh\n' "$name" "$name"
