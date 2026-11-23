#!/usr/bin/env python3
"""CI-safe evaluation integrity checks.

Validates only files tracked in evals/: no model calls, no network, no
dependency on the ignored data/ directory. The real quality gate needs a frozen
report and therefore runs explicitly, not in CI.

检索集分两层检查：历史版本（已被后续版本 ``supersedes`` 的）只查 id 与题目是否为空，
当前版本才要求每条题要么可判分、要么写清"为什么不可判分、由谁接管"。理由里出现
"为了让/使…成立"这类表述直接拒绝——不计分的依据只能来自材料本身的客观属性，
见 ``docs/evaluation-integrity.md``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
# 检索题标记"不可文字检索"时必须给客观依据和接管者；理由里出现这类表述，
# 说明改动是为了让某个结论成立，而不是因为材料本身如此。
INTENT_PHRASES = ("为了让", "为了保", "为了符", "使结论", "让结论", "结论成立", "结论继续")
MIN_REASON_CHARS = 6


def load_all(base: Path) -> list[tuple[str, Path, dict | None]]:
    """读出 evals/ 下全部 JSON，返回 (相对路径, 路径, 对象或 None)。"""
    loaded: list[tuple[str, Path, dict | None]] = []
    for path in sorted((base / "evals").glob("*.json")):
        label = path.relative_to(base).as_posix()
        try:
            loaded.append((label, path, json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError) as error:
            loaded.append((label, path, None))
            print(f"评测完整性检查：{label} 无法解析：{error}")
    return loaded


def check_suite(label: str, data: dict, errors: list[str]) -> None:
    if not data.get("id") or not data.get("version"):
        errors.append(f"{label}: 任务集缺少 id 或 version")
    if not data.get("rubric_version"):
        errors.append(f"{label}: 任务集缺少 rubric_version，配对对比无法判定可比性")
    case_ids: set[str] = set()
    for case in data.get("cases", []):
        if not isinstance(case, dict):
            errors.append(f"{label}: cases 里有一项不是对象")
            continue
        cid = case.get("id")
        if not cid or cid in case_ids:
            errors.append(f"{label}: case id 缺失或重复：{cid!r}")
        case_ids.add(str(cid))
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
            turn_ids.add(str(tid))
            if not isinstance(turn.get("input"), dict) or not turn.get("input", {}).get("question"):
                errors.append(f"{label}: {cid}:{tid} 缺少 input.question")
            gold = turn.get("gold")
            if not isinstance(gold, dict) or not gold:
                errors.append(f"{label}: {cid}:{tid} 缺少 gold，无法评测")
    if not any(c.get("split") == "dev" for c in data.get("cases", []) if isinstance(c, dict)):
        errors.append(f"{label}: 没有 dev 回合，无法做默认试运行")


def check_baseline(label: str, data: dict, errors: list[str]) -> None:
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


def check_retrieval(label: str, data: dict, errors: list[str], strict: bool) -> None:
    """检索集：当前版本每题要能判分，或写明为什么不可判分、由谁接管。"""
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        errors.append(f"{label}: 检索查询集为空")
        return
    query_ids: set[str] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            errors.append(f"{label}: 第 {index + 1} 条查询不是对象")
            continue
        qid = query.get("id")
        name = f"{label}:{qid or f'#{index + 1}'}"
        if not qid or qid in query_ids:
            errors.append(f"{name}: 查询 id 缺失或重复：{qid!r}")
        query_ids.add(str(qid))
        if not strict:
            if not query.get("question"):
                errors.append(f"{name}: 缺少 question")
            continue

        if not query.get("modality"):
            errors.append(f"{name}: 缺少 modality，无法按模态统计")
        if not query.get("question"):
            errors.append(f"{name}: 缺少 question")

        unreachable = query.get("unretrievable")
        # 负例（预期检索不到任何目标）要么标 negative，要么写明标注依据。
        annotated_negative = query.get("negative") is True or bool(
            str(query.get("annotation_evidence") or "").strip()
        )
        if not query.get("expected") and not annotated_negative and unreachable is None:
            errors.append(
                f"{name}: expected 为空，既不能判分也不能说明不可达"
                "（负例标 negative: true 或写 annotation_evidence，不可文字检索写 unretrievable）"
            )
        if unreachable is None:
            continue
        if not isinstance(unreachable, dict):
            errors.append(f"{name}: unretrievable 应为对象，说明依据与接管者")
            continue
        reason = str(unreachable.get("reason") or "").strip()
        if len(reason) < MIN_REASON_CHARS:
            errors.append(f"{name}: unretrievable.reason 太短，写清材料本身的客观属性")
        for phrase in INTENT_PHRASES:
            if phrase in reason:
                errors.append(
                    f"{name}: unretrievable.reason 含意图性表述 {phrase!r}——"
                    "不计分的理由只能来自材料属性，见 docs/evaluation-integrity.md"
                )
        if not str(unreachable.get("readable_by") or "").strip():
            errors.append(f"{name}: 不计分必须写明 readable_by（谁来接管这道题）")


def check_reviewer_calibration(label: str, data: dict, errors: list[str]) -> None:
    if not data.get("reviewer_kind"):
        errors.append(f"{label}: 裁判校准缺少 reviewer_kind")
    if not data.get("samples"):
        errors.append(f"{label}: 裁判校准没有样本")


def main(root: Path | None = None) -> int:
    base = Path(root) if root is not None else ROOT
    loaded = load_all(base)
    if not loaded:
        print("评测完整性检查失败：evals/ 下没有 JSON 文件")
        return 1
    unparsable = [label for label, _, data in loaded if data is None]
    errors: list[str] = [f"{label}: 无法解析 JSON" for label in unparsable]

    # 被后续版本 supersede 的检索集属历史，只做基本检查。
    superseded = {
        data.get("supersedes")
        for _, _, data in loaded
        if isinstance(data, dict) and data.get("supersedes") is not None
    }
    docs = [(label, data) for label, _, data in loaded if isinstance(data, dict)]

    suites = baselines = retrievals = 0
    for label, data in docs:
        if "cases" in data:
            suites += 1
            check_suite(label, data, errors)
        elif "thresholds" in data and "metrics" in data:
            baselines += 1
            check_baseline(label, data, errors)
        elif "samples" in data:
            check_reviewer_calibration(label, data, errors)
        elif "queries" in data:
            retrievals += 1
            check_retrieval(label, data, errors, strict=data.get("version") not in superseded)

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
        f"评测完整性检查通过：{len(loaded)} 个文件、{suites} 个任务集、"
        f"{baselines} 个质量基线、{retrievals} 个检索集，无模型调用。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
