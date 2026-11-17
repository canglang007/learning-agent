"""公开检查脚本本身的回归：它漏判一次，就等于密钥或地址已经公开。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("check_public", ROOT / "scripts/check_public.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = load()


def scan(line):
    return CHECK.line_findings("demo.md", 1, line)


def test_public_ip_is_flagged_but_local_and_example_ranges_are_not():
    assert scan("地址：http://20.205.19.150:8320")
    assert not scan("监听 0.0.0.0:8320")
    assert not scan("本机 http://127.0.0.1:8320")
    assert not scan("内网 192.168.1.10 或 10.0.0.5")
    assert not scan("文档示例 203.0.113.9")  # TEST-NET-3，专门用于文档
    assert not scan("连接 8.8.8.8 测出网")  # 公开 DNS，正常用法


def test_access_token_assignment_is_flagged_but_placeholders_are_not():
    assert scan("LA_ACCESS_TOKEN=GdrXWV3HCE4N")
    assert not scan("LA_ACCESS_TOKEN=")
    assert not scan("LA_ACCESS_TOKEN=这里填访问口令")
    assert not scan("LA_ACCESS_TOKEN=<访问口令>")
    assert not scan("LA_ACCESS_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(9))')")


def test_personal_home_path_is_flagged():
    assert scan("/Users/canglang007/Desktop/work")
    assert scan("/home/alice/project")
    assert not scan("/home/user/project")
    assert not scan("/Users/you/Desktop")


def test_api_key_is_flagged():
    assert scan("LA_API_KEY=sk-abcdefghijklmnopqrstuvwxyz")
    assert not scan("LA_API_KEY=your-key")
