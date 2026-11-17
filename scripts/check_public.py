#!/usr/bin/env python3
"""Fail when tracked files contain likely secrets or private runtime data."""

from __future__ import annotations

import ipaddress
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024

FORBIDDEN_PREFIXES = ("data/", "notes/", "experiments/", "output/", ".venv/")
FORBIDDEN_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".pem", ".key")
FORBIDDEN_NAMES = {".env"}

# 公开 DNS/连通性探测地址是正常用法，不算泄漏。
ALLOWED_IPS = {"8.8.8.8", "1.1.1.1", "223.5.5.5", "114.114.114.114"}
# 文档里的占位用户目录，不指向某个人的机器。
ALLOWED_HOME_NAMES = {"user", "you", "your", "example", "runner", "app", "sandbox", "shared"}

PATTERNS = (
    ("openai-style key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("github token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("github fine-grained token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("aws access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("LA_API_KEY assignment", re.compile(r"LA_API_KEY\s*=\s*[\"']?([^\s\"']{8,})")),
    ("LA_ASR_API_KEY assignment", re.compile(r"LA_ASR_API_KEY\s*=\s*[\"']?([^\s\"']{8,})")),
    (
        "LA_EMBEDDING_API_KEY assignment",
        re.compile(r"LA_EMBEDDING_API_KEY\s*=\s*[\"']?([^\s\"']{8,})"),
    ),
    (
        "LA_ACCESS_TOKEN assignment",
        re.compile(r"LA_ACCESS_TOKEN\s*=\s*[\"']?([^\s\"'#]{6,})"),
    ),
    (
        "ANTHROPIC_AUTH_TOKEN assignment",
        re.compile(r"ANTHROPIC_AUTH_TOKEN\s*=\s*[\"']?([^\s\"']{8,})"),
    ),
)
PLACEHOLDER = re.compile(
    r"^(your|test|example|placeholder|changeme|none|empty|<)", re.IGNORECASE
)
# 文档里的中文占位说明，例如“这里填访问口令”。
PLACEHOLDER_HINTS = ("你的", "你的密钥", "这里", "填", "口令", "密钥", "示例")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
HOME_PATH = re.compile(r"/(?:Users|home)/([A-Za-z0-9._-]+)")


# 检查器自己的回归用例必须写出"坏样本"，否则测不出漏判。只豁免这一个文件，
# 其他文件照扫；往这里塞真密钥和往别处塞一样明显。
FIXTURES = {"tests/test_check_public.py"}


def exempt(relative: str) -> bool:
    return relative in FIXTURES


def finding_for(label: str, match: re.Match[str]) -> str | None:
    if match.lastindex:
        value = match.group(1)
        if PLACEHOLDER.search(value) or "..." in value:
            return None
        # shell 命令替换（生成口令的推荐写法）不是密钥本身。
        if value.startswith(("$", "(", "`")):
            return None
        if any(hint in value for hint in PLACEHOLDER_HINTS):
            return None
    return label


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / name for name in result.stdout.decode().split("\0") if name]


def line_findings(relative: str, line_number: int, line: str) -> list[str]:
    """扫描一行，返回可直接打印的结论。抽成函数是为了能被测试覆盖。"""
    found: list[str] = []
    for label, pattern in PATTERNS:
        match = pattern.search(line)
        if match:
            finding = finding_for(label, match)
            if finding:
                found.append(f"{relative}:{line_number}: 疑似 {finding}")
    for candidate in IPV4.finditer(line):
        address = candidate.group()
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_global and address not in ALLOWED_IPS:
            found.append(f"{relative}:{line_number}: 公网地址 {address}，公开仓库会暴露部署实例")
    for match in HOME_PATH.finditer(line):
        if match.group(1).lower() not in ALLOWED_HOME_NAMES:
            found.append(f"{relative}:{line_number}: 个人目录 {match.group()}，公开仓库会暴露本机用户名")
    return found


def main() -> int:
    try:
        files = tracked_files()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"无法读取 git 跟踪文件：{error}", file=sys.stderr)
        return 2

    findings: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        env_file = path.name.startswith(".env.") and path.name != ".env.example"
        if path.name in FORBIDDEN_NAMES or env_file:
            findings.append(f"{relative}: 运行时配置文件不应被跟踪")
            continue
        if relative.startswith(FORBIDDEN_PREFIXES) or relative.endswith(FORBIDDEN_SUFFIXES):
            findings.append(f"{relative}: 私有数据或数据库不应被跟踪")
            continue
        if exempt(relative):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            findings.append(f"{relative}: 无法读取")
            continue
        if size > MAX_BYTES:
            findings.append(f"{relative}: 文件超过 10 MiB，确认是否为私有数据")
            continue
        # .env.example 也要扫：示例文件里出现真实密钥同样是泄漏
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            findings.extend(line_findings(relative, line_number, line))

    if findings:
        print("公开检查失败：", file=sys.stderr)
        for finding in findings:
            print(" - " + finding, file=sys.stderr)
        return 1
    print(f"公开检查通过：{len(files)} 个跟踪文件，未发现私有数据或疑似密钥。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
