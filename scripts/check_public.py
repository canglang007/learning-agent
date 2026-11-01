#!/usr/bin/env python3
"""Fail when tracked files contain likely secrets or private runtime data."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024

FORBIDDEN_PREFIXES = ("data/", "notes/", "experiments/", "output/", ".venv/")
FORBIDDEN_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".pem", ".key")
FORBIDDEN_NAMES = {".env"}

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
        "ANTHROPIC_AUTH_TOKEN assignment",
        re.compile(r"ANTHROPIC_AUTH_TOKEN\s*=\s*[\"']?([^\s\"']{8,})"),
    ),
)
PLACEHOLDER = re.compile(r"^(your|test|example|placeholder|changeme|none|empty|<)", re.IGNORECASE)


def finding_for(label: str, match: re.Match[str]) -> str | None:
    if match.lastindex:
        value = match.group(1)
        if PLACEHOLDER.search(value) or "..." in value:
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
        try:
            size = path.stat().st_size
        except OSError:
            findings.append(f"{relative}: 无法读取")
            continue
        if size > MAX_BYTES:
            findings.append(f"{relative}: 文件超过 10 MiB，确认是否为私有数据")
            continue
        if relative == ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS:
                match = pattern.search(line)
                if match:
                    finding = finding_for(label, match)
                    if finding:
                        findings.append(f"{relative}:{line_number}: 疑似 {finding}")

    if findings:
        print("公开检查失败：", file=sys.stderr)
        for finding in findings:
            print(" - " + finding, file=sys.stderr)
        return 1
    print(f"公开检查通过：{len(files)} 个跟踪文件，未发现私有数据或疑似密钥。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
