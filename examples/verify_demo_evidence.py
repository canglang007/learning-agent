"""Check that the local, ignored demo evidence bundle is present."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "browser_acceptance": ROOT / "data/demo/browser-acceptance.json",
    "comparison": ROOT / "data/evaluations/evaluation-pair-v1/comparison.json",
    "component": ROOT / "data/evaluations/evaluation-pair-v1/component-output.json",
    "report": ROOT / "data/evaluations/evaluation-pair-v1/report.json",
}


def main():
    missing = [name for name, path in FILES.items() if not path.is_file()]
    if missing:
        raise SystemExit("缺少本地真实证据：" + ", ".join(missing))
    payload = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    component = payload["component"].get("results", [{}])[0]
    print("演示证据文件齐全")
    print("组件检索：", {key: component.get(key) for key in ("annotated_hit_at_5", "annotated_mrr_at_5")})
    print("浏览器验收：", payload["browser_acceptance"].get("turns", "已记录"))
    print("提示：这些是本地真实运行汇总，不代表独立真人金标准。")


if __name__ == "__main__":
    main()
