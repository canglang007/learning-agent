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


def newest_run():
    """每次真实重跑都会写一个新文件；检查最新的一份，而不是钉死某个日期。"""
    runs = sorted((ROOT / "data/demo").glob("multimodal-run-*.json"))
    return runs[-1] if runs else None


def main():
    run_path = newest_run()
    missing = [name for name, path in FILES.items() if not path.is_file()]
    if run_path is None:
        missing.append("multimodal_run")
    if missing:
        raise SystemExit("缺少本地真实证据：" + ", ".join(missing))
    payload = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    component = payload["component"].get("results", [{}])[0]
    run = json.loads(run_path.read_text())
    print("演示证据文件齐全")
    print("组件检索：", {key: component.get(key) for key in ("annotated_hit_at_5", "annotated_mrr_at_5")})
    print("浏览器验收：", payload["browser_acceptance"].get("turns", "已记录"))
    print("最新真实多模态运行：", {"file": run_path.name, "state": run.get("state"), "route": (run.get("route") or {}).get("route") if isinstance(run.get("route"), dict) else run.get("route")})
    print(
        "工具序列：",
        {
            "calls": len(run.get("tool_steps", [])),
            "failures": run.get("tool_failures"),
            "references": len(run.get("references", [])),
            "images": (run.get("last_step_estimate") or {}).get("images"),
        },
    )
    print("提示：这些是本地真实运行汇总，不代表独立真人金标准。")


if __name__ == "__main__":
    main()
