import json
import statistics
from pathlib import Path

from learning_agent.evaluation.graders import grade
from learning_agent.evaluation.ledger import Ledger, save_json


def report(root, *, include_content=False, readonly=False):
    root = Path(root)
    ledger = Ledger(root / "experiment.db", readonly=readonly)
    definition = ledger.get("definition", {})
    corpus = ledger.get("corpus", {})
    rows = []
    usage = {}
    for trial in ledger.records("trial:"):
        for spec in trial["case"]["turns"]:
            item = trial["turns"].get(spec["id"], {"status": "not_started"})
            trace = item.get("trace", {})
            assessed = (
                grade(trace, spec["gold"], corpus.get("assets", {}))
                if trace
                else {"checks": [], "findings": []}
            )
            reviews = ledger.records(f"review:{trial['id']}:{spec['id']}:")
            row = {
                "trial_id": trial["id"],
                "variant": trial["variant"],
                "case_id": trial["case"]["id"],
                "family": trial["case"]["family"],
                "turn_id": spec["id"],
                "split": trial["case"]["split"],
                "repeat": trial["repeat"],
                "status": "budget_exhausted"
                if trace.get("state") == "budget_exhausted"
                else item["status"],
                "runtime_state": trace.get("state"),
                "latency": trace.get("latency", {}),
                "accepted_s": item.get("accepted_s"),
                "timing_comparable": not any(
                    e["type"] in {"run.interrupted", "run.resumed"} for e in trace.get("events", [])
                ),
                "profile": trial.get("profile"),
                "checks": assessed["checks"],
                "findings": assessed["findings"],
                "review": reviews[-1] if reviews else None,
                "review_count": len(reviews),
                "operations": [],
            }
            for op in trace.get("operations", []):
                row["operations"].append(
                    {k: op[k] for k in ("id", "purpose", "attempt", "state", "usage_status")}
                )
                bucket = usage.setdefault(
                    op["purpose"],
                    {"attempts": 0, "known": 0, "unknown": 0, "unavailable": 0, "tokens": {}},
                )
                bucket["attempts"] += 1
                bucket[op["usage_status"]] += 1
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                ):
                    value = (op.get("usage") or {}).get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        bucket["tokens"][key] = bucket["tokens"].get(key, 0) + value
            if include_content:
                row.update(
                    answer=trace.get("answer"),
                    question=spec["input"]["question"],
                    gold=spec["gold"],
                    trace=trace,
                    assets=corpus.get("assets", {}),
                )
            if row["review"]:
                row["findings"].extend(row["review"].get("findings", []))
            rows.append(row)
    judge_usage = ledger.records("judge:") + ledger.records("judge-history:")
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    wall = [r["latency"]["wall_s"] for r in rows if "wall_s" in r["latency"]]
    planned = sum(len(t["case"]["turns"]) for t in definition.get("trials", []))
    reviewed = [r for r in rows if r["review"] and r["review"].get("status") == "complete"]
    result = {
        "schema_version": 1,
        "manifest": ledger.get("manifest"),
        "planned": planned,
        "recorded": len(rows),
        "counts": counts,
        "unstarted_trials_turns": planned - len(rows),
        "reviewed": len(reviewed),
        "review_coverage": len(reviewed) / planned if planned else None,
        "usage": usage,
        "reservations": ledger.usage(),
        "judge_operations": judge_usage,
        "median_wall_s": statistics.median(wall) if wall else None,
        "interrupted_turns": sum(not r["timing_comparable"] for r in rows),
        "wall_range_s": [min(wall), max(wall)] if wall else None,
        "preparation_error": ledger.get("preparation_error"),
        "rows": rows,
        "note": "执行状态、规则检查和内容复核分开；unknown/unavailable不视为零费用。",
    }
    if not include_content and not readonly:
        save_json(root / "report.json", result)
        (root / "report.md").write_text(
            f"# 评测摘要\n\n更新时间：{(ledger.get('manifest') or {}).get('created_at', 'unknown')}（UTC）。\n\n计划{planned}回合；状态{json.dumps(counts, ensure_ascii=False)}；内容复核{len(reviewed)}/{planned}。\n\n中位墙钟：{result['median_wall_s']}秒。usage与缺失项见report.json。\n"
        )
    return result


def compare(left, right=None, *, variants=None):
    a, b = report(left), report(right or left)

    def key(row):
        return row["case_id"], row["turn_id"], row["repeat"]

    if variants:
        a["rows"] = [r for r in a["rows"] if r["variant"] == variants[0]]
        b["rows"] = [r for r in b["rows"] if r["variant"] == variants[1]]
    pairs = []
    rb = {key(r): r for r in b["rows"]}
    if right is None and not variants:
        raise ValueError("同实验对比需要指定两个--variants")
    left_def = Ledger(Path(left) / "experiment.db").get("definition", {})
    right_def = Ledger(Path(right or left) / "experiment.db").get("definition", {})
    corpus_a = Ledger(Path(left) / "experiment.db").get("corpus", {})
    corpus_b = Ledger(Path(right or left) / "experiment.db").get("corpus", {})
    same_assets = corpus_a == corpus_b
    if Path(left).resolve() != Path(right or left).resolve():

        def sources(corpus):
            return {
                k: {
                    f: v.get(f)
                    for f in (
                        "name",
                        "source_start_seconds",
                        "revision",
                        "source_bytes",
                        "source_modified_ns",
                    )
                }
                for k, v in corpus.get("assets", {}).items()
            }

        same_assets = sources(corpus_a) == sources(corpus_b) and all(
            "source_bytes" in v for v in corpus_a.get("assets", {}).values()
        )
    same_tasks = left_def.get("suite") == right_def.get("suite")
    for row in a["rows"]:
        other = rb.get(key(row))
        status = "needs_review"
        reason = "内容复核尚未完成"
        if not other:
            status, reason = "incomparable", "另一配置缺少同任务回合"
        elif (
            not same_assets
            or not same_tasks
            or a["manifest"]["suite_version"] != b["manifest"]["suite_version"]
            or a["manifest"]["rubric_version"] != b["manifest"]["rubric_version"]
        ):
            status, reason = "incomparable", "任务、语料或rubric不一致/缺少可比信息"
        elif (
            row["review"]
            and other["review"]
            and all(r["review"].get("status") == "complete" for r in (row, other))
        ):
            old, new = row["review"].get("criteria", {}), other["review"].get("criteria", {})
            if (
                row["review"]["reviewer_kind"] != other["review"]["reviewer_kind"]
                or row["review"]["reviewer"] != other["review"]["reviewer"]
            ):
                reason = "复核来源不同，需要统一口径重评"
            elif set(old) == set(new) and old:
                if "needs_review" in [*old.values(), *new.values()]:
                    reason = "存在未决条件，不能把不确定性当作质量升降"
                else:
                    order = {"fail": 0, "pass": 1}
                    deltas = [order[new[k]] - order[old[k]] for k in old]
                    deltas.append(
                        int(row["review"]["serious_error"]) - int(other["review"]["serious_error"])
                    )
                    status = (
                        "unchanged"
                        if not any(deltas)
                        else "improved"
                        if min(deltas) >= 0
                        else "regressed"
                        if max(deltas) <= 0
                        else "mixed"
                    )
                    reason = "逐条件配对；重大错误单列，不以平均分抵消退化"
        pairs.append(
            {
                "case_id": row["case_id"],
                "turn_id": row["turn_id"],
                "repeat": row["repeat"],
                "result": status,
                "reason": reason,
                "timing_comparable": row["timing_comparable"]
                and bool(other and other["timing_comparable"]),
                "before_state": row["status"],
                "before_review": row.get("review"),
                "before_checks": row.get("checks"),
                "after_checks": other.get("checks") if other else None,
                "after_review": other.get("review") if other else None,
                "after_state": other["status"] if other else None,
                "before_latency": row["latency"],
                "after_latency": other["latency"] if other else None,
            }
        )
    result = {
        "pairs": pairs,
        "note": "同Case回合相关；实际降级、配置与样本量须一起解释，不自动声称因果或显著性。",
    }
    save_json(Path(right or left) / "comparison.json", result)
    return result
