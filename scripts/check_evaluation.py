#!/usr/bin/env python3
"""CI-safe evaluation integrity checks.

Validates only files tracked in evals/: no model calls, no network, no
dependency on the ignored data/ directory. The real quality gate needs a frozen
report and therefore runs explicitly, not in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
REQUIRED_METRICS = (
    "planned",
    "recorded",
    "delivered",
    "completed",
    "evidence_pass",
    "evidence_total",
    "reviewed",
    "content_pass",
    "serious_errors",
)


def main() -> int:
    errors: list[str] = []
    paths = sorted(EVALS.glob("*.json"))
    if not paths:
        print("评测完整性检查失败：evals/ 下没有 JSON 文件")
        return 1

    suites = baselines = 0
    case_ids: set[str] = set()
    for path in paths:
        label = path.relative_to(ROOT).as_posix()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{label}: 无法解析 JSON：{error}")
            continue

        if "cases" in data:
            suites += 1
            if not data.get("id") or not data.get("version"):
                errors.append(f"{label}: 任务集缺少 id 或 version")
            if not data.get("rubric_version"):
                errors.append(f"{label}: 任务集缺少 rubric_version，配对对比无法判定可比性")
            for case in data.get("cases", []):
                cid = case.get("id")
                if not cid or cid in case_ids:
                    errors.append(f"{label}: case id 缺失或重复：{cid!r}")
                case_ids.add(cid)
                if case.get("split") not in {"dev", "holdout", "all"}:
                    errors.append(f"{label}: {cid} 的 split 无效：{case.get('split')!r}")
                turns = case.get("turns") or []
                if not turns:
                    errors.append(f"{label}: {cid} 没有回合")
                turn_ids: set[str] = set()
                for turn in turns:
                    tid = turn.get("id")
                    if not tid or tid in turn_ids:
                        errors.append(f"{label}: {cid} 的 turn id 缺失或重复：{tid!r}")
                    turn_ids.add(tid)
                    if not isinstance(turn.get("input"), dict) or not turn.get("input", {}).get(
                        "question"
                    ):
                        errors.append(f"{label}: {cid}:{tid} 缺少 input.question")
                    gold = turn.get("gold")
                    if not isinstance(gold, dict) or not gold:
                        errors.append(f"{label}: {cid}:{tid} 缺少 gold，无法评测")
            if not any(c.get("split") == "dev" for c in data.get("cases", [])):
                errors.append(f"{label}: 没有 dev 回合，无法做默认试运行")

        elif "thresholds" in data and "metrics" in data:
            baselines += 1
            from learning_agent.evaluation.gate import BASELINE_SCHEMA

            if data.get("schema_version") != BASELINE_SCHEMA:
                errors.append(
                    f"{label}: schema_version={data.get('schema_version')!r}，"
                    f"当前 gate 需要 {BASELINE_SCHEMA}"
                )
            if not (data.get("source") or {}).get("experiment"):
                errors.append(f"{label}: 缺少 source.experiment，基线来源不可追溯")
            scope = data.get("scope") or {}
            if not scope.get("turns"):
                errors.append(f"{label}: scope.turns 为空，无法判定可比性")
            metrics = data.get("metrics") or {}
            for key in REQUIRED_METRICS:
                value = metrics.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"{label}: metrics.{key} 必须是非负整数，实际 {value!r}")
            if metrics.get("reviewed", 0) > metrics.get("recorded", 0):
                errors.append(f"{label}: reviewed 不能大于 recorded")
            if metrics.get("evidence_pass", 0) > metrics.get("evidence_total", 0):
                errors.append(f"{label}: evidence_pass 不能大于 evidence_total")
            thresholds = data.get("thresholds") or {}
            coverage = thresholds.get("min_review_coverage")
            if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
                errors.append(f"{label}: min_review_coverage 必须在 0..1")

        elif "samples" in data:
            if not data.get("reviewer_kind"):
                errors.append(f"{label}: 裁判校准缺少 reviewer_kind")
            if not data.get("samples"):
                errors.append(f"{label}: 裁判校准没有样本")

        elif "queries" in data:
            if not data.get("queries"):
                errors.append(f"{label}: 检索查询集为空")
        elif not isinstance(data, dict):
            errors.append(f"{label}: 顶层应为对象")

    if not suites:
        errors.append("没有找到任何任务集（含 cases 的 JSON）")
    if not baselines:
        errors.append("没有找到质量基线快照（含 metrics/thresholds 的 JSON）")

    if errors:
        print(f"评测完整性检查失败：{len(errors)} 项")
        for item in errors:
            print(f"  - {item}")
        return 1
    print(
        f"评测完整性检查通过：{len(paths)} 个文件、{suites} 个任务集、"
        f"{baselines} 个质量基线，无模型调用。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
