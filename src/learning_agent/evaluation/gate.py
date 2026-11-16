"""Quality gate for the evaluation loop.

Reads reports that a previous evaluation already froze and compares them against
an accepted baseline snapshot. This module never calls a model and never writes
to an experiment database: a gate must be reproducible offline and cheap enough
to run on every change.

The gate is deliberately conservative:
  * a different suite/rubric/variant/case set is ``incomparable``, never a pass;
  * missing content review is ``insufficient_evidence``, never a pass;
  * a drop in delivery, evidence or reviewed content is a regression;
  * latency and token counts are reported but never used as quality signals.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from learning_agent.evaluation.ledger import save_json

BASELINE_SCHEMA = 1
HARD_METRICS = ("delivered", "evidence_pass", "content_pass")
EXIT_CODES = {
    "passed": 0,
    "regressed": 1,
    "incomparable": 2,
    "insufficient_evidence": 3,
}


def read_report(experiment: Path) -> dict:
    path = Path(experiment) / "report.json"
    if not path.is_file():
        raise ValueError(
            f"缺少 {path}；先运行 python -m learning_agent.evaluation report {experiment}"
        )
    return json.loads(path.read_text())


def _content_pass(review: dict | None) -> bool:
    if not review or review.get("status") != "complete":
        return False
    if review.get("serious_error"):
        return False
    criteria = review.get("criteria") or {}
    return bool(criteria) and all(value == "pass" for value in criteria.values())


def summarize(experiment: Path, *, variant: str | None = None) -> dict:
    """Reduce a frozen report to comparable quality metrics and its scope."""
    data = read_report(experiment)
    rows = [row for row in data.get("rows", []) if variant is None or row.get("variant") == variant]
    if not rows:
        raise ValueError(f"{experiment} 中没有 variant={variant!r} 的回合")

    checks = [c for row in rows for c in row.get("checks", [])]
    evidence = [c for c in checks if str(c.get("name", "")).startswith("evidence_")]
    reviews = [row.get("review") for row in rows if _content_pass(row.get("review"))]
    counted = [row.get("review") for row in rows if (row.get("review") or {}).get("status") == "complete"]
    manifest = data.get("manifest") or {}
    planned = sum(
        len(t["case"]["turns"]) for t in (data.get("manifest") or {}).get("trials", [])
    ) or len(rows)
    delivered = sum(
        row.get("status") in {"completed", "budget_exhausted"}
        or row.get("runtime_state") in {"completed", "budget_exhausted"}
        for row in rows
    )

    return {
        "scope": {
            "suite_version": manifest.get("suite_version"),
            "rubric_version": manifest.get("rubric_version"),
            "variant": variant,
            "cases": sorted({row["case_id"] for row in rows}),
            "turns": sorted({f"{row['case_id']}:{row['turn_id']}" for row in rows}),
        },
        "metrics": {
            "planned": planned,
            "recorded": len(rows),
            "delivered": delivered,
            "completed": sum(row.get("status") == "completed" for row in rows),
            "evidence_pass": sum(c.get("status") == "pass" for c in evidence),
            "evidence_total": len(evidence),
            "evidence_needs_review": sum(c.get("status") == "needs_review" for c in evidence),
            "reviewed": len(counted),
            "content_pass": len(reviews),
            "serious_errors": sum(int(r.get("serious_error") or 0) for r in counted),
        },
        "cost_context": {
            "main_model_attempts": (data.get("usage") or {})
            .get("main_model", {})
            .get("attempts", 0),
            "main_model_tokens": (data.get("usage") or {})
            .get("main_model", {})
            .get("tokens", {})
            .get("total_tokens", 0),
            "median_wall_s": data.get("median_wall_s"),
        },
    }


def make_baseline(
    experiment: Path,
    *,
    variant: str | None = None,
    source: str | None = None,
    note: str = "",
) -> dict:
    """Freeze an accepted report as the quality baseline. Review must be complete."""
    summary = summarize(experiment, variant=variant)
    metrics = summary["metrics"]
    if not metrics["reviewed"]:
        raise ValueError("基线必须包含已完成的内容复核；未复核的报告不能当质量水位")
    return {
        "schema_version": BASELINE_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {"experiment": source or str(experiment), "variant": variant},
        "scope": summary["scope"],
        "metrics": metrics,
        "thresholds": {
            "min_review_coverage": 1.0,
            "max_serious_errors": metrics["serious_errors"],
        },
        "note": note or "基线来自一次真实评测；样本量有限，只作回归信号，不作显著性结论。",
    }


def evaluate(baseline: dict, candidate: dict) -> dict:
    """Compare a candidate summary against a baseline. No network, no side effects."""
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        raise ValueError(f"基线 schema_version 应为 {BASELINE_SCHEMA}")

    scope_deltas = {
        key: {"baseline": baseline["scope"].get(key), "candidate": candidate["scope"].get(key)}
        for key in ("suite_version", "rubric_version", "variant", "cases", "turns")
        if baseline["scope"].get(key) != candidate["scope"].get(key)
    }
    if scope_deltas:
        return {
            "result": "incomparable",
            "reason": "任务集、rubric、变体或回合范围不同；不能据此判定质量升降。",
            "scope_deltas": scope_deltas,
            "metric_deltas": {},
            "cost_context": candidate.get("cost_context", {}),
        }

    b, c = baseline["metrics"], candidate["metrics"]
    deltas = {key: c[key] - b[key] for key in b if key in c}
    coverage = c["reviewed"] / c["recorded"] if c["recorded"] else 0
    minimum = baseline.get("thresholds", {}).get("min_review_coverage", 1.0)
    if coverage < minimum or c["reviewed"] < b["reviewed"]:
        return {
            "result": "insufficient_evidence",
            "reason": f"内容复核 {c['reviewed']}/{c['recorded']}（基线 {b['reviewed']}，要求覆盖率 ≥{minimum}）；未复核不算通过。",
            "scope_deltas": {},
            "metric_deltas": deltas,
            "cost_context": candidate.get("cost_context", {}),
        }

    regressions = [
        {"metric": key, "baseline": b[key], "candidate": c[key], "delta": deltas[key]}
        for key in HARD_METRICS
        if deltas[key] < 0
    ]
    max_errors = baseline.get("thresholds", {}).get("max_serious_errors", b["serious_errors"])
    if c["serious_errors"] > max_errors:
        regressions.append(
            {
                "metric": "serious_errors",
                "baseline": b["serious_errors"],
                "candidate": c["serious_errors"],
                "delta": deltas["serious_errors"],
            }
        )
    if regressions:
        return {
            "result": "regressed",
            "reason": "交付、证据命中或已复核内容出现退化，或重大错误增加；不以平均分抵消。",
            "regressions": regressions,
            "scope_deltas": {},
            "metric_deltas": deltas,
            "cost_context": candidate.get("cost_context", {}),
        }

    def improved(key: str, delta: int) -> bool:
        return delta < 0 if key == "serious_errors" else delta > 0

    improvements = [
        {"metric": key, "baseline": b[key], "candidate": c[key], "delta": deltas[key]}
        for key in HARD_METRICS + ("serious_errors",)
        if improved(key, deltas[key])
    ]
    return {
        "result": "passed",
        "reason": "范围一致、复核充分，硬指标未退化。",
        "improvements": improvements,
        "scope_deltas": {},
        "metric_deltas": deltas,
        "cost_context": candidate.get("cost_context", {}),
    }


def gate(baseline_path: Path, experiment: Path, *, variant: str | None = None) -> dict:
    baseline = json.loads(Path(baseline_path).read_text())
    candidate = summarize(experiment, variant=variant)
    result = evaluate(baseline, candidate)
    result["baseline"] = str(baseline_path)
    result["experiment"] = str(experiment)
    result["note"] = (
        "质量门禁只比较已冻结的证据；样本量与复核来源须一并解释，token/耗时不参与质量判定。"
    )
    return result


def write_baseline(path: Path, baseline: dict) -> None:
    save_json(path, baseline)
