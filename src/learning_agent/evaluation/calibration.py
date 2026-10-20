"""Judge calibration on explicitly authored answers, not Agent teaching trials."""

import json
from pathlib import Path

from learning_agent.evaluation.ledger import Ledger, save_json
from learning_agent.evaluation.report import report
from learning_agent.evaluation.review import judge
from learning_agent.run_store import now


async def calibrate(samples_path, output, settings, *, provider=None):
    output = Path(output)
    if output.exists():
        raise ValueError("校准输出必须是新目录，避免无意重复付费")
    samples = json.loads(Path(samples_path).read_text())
    ledger = Ledger(output / "experiment.db")
    ledger.put("limits", {"judge": len(samples["samples"])})
    ledger.put(
        "manifest",
        {
            "id": output.name,
            "kind": "authored_judge_calibration",
            "created_at": now(),
            "suite_id": samples["id"],
            "suite_version": samples["version"],
            "rubric_version": "1",
            "reference_reviewer": "developer_agent",
            "agent_calls": 0,
        },
    )
    definitions = []
    for i, sample in enumerate(samples["samples"]):
        tid = f"authored-{i}"
        turn = {
            "id": "answer",
            "input": {"question": sample["question"]},
            "gold": {"required_concepts": [sample["criterion"]]},
            "actions": [],
        }
        case = {"id": tid, "family": tid, "split": "dev", "turns": [turn]}
        definitions.append({"id": tid, "case": case})
        trace = {
            "schema_version": 1,
            "kind": "authored_fixture_not_agent_execution",
            "run_id": tid,
            "revision": 0,
            "state": "completed",
            "complete": True,
            "messages": [{"role": "assistant", "kind": "answer", "content": sample["answer"]}],
            "answer": sample["answer"],
            "events": [],
            "operations": [],
            "tools": [],
            "latency": {},
        }
        ledger.put(
            "trial:" + tid,
            {
                "id": tid,
                "variant": "authored",
                "case": case,
                "repeat": 0,
                "turns": {"answer": {"status": "completed", "trace": trace}},
            },
        )
    ledger.put("definition", {"trials": definitions})
    await judge(output, settings, limit=len(samples["samples"]), provider=provider)
    rows = report(output)["rows"]
    comparisons = []
    for row, sample in zip(rows, samples["samples"], strict=True):
        actual = (row.get("review") or {}).get("criteria", {}).get("concept_0")
        comparisons.append(
            {
                "id": row["trial_id"],
                "expected": sample["expected"],
                "actual": actual,
                "agrees": actual == sample["expected"],
                "reference_reason": sample["reason"],
            }
        )
    result = {
        "kind": "judge_calibration",
        "reference_reviewer": "developer_agent",
        "samples": len(comparisons),
        "agrees": sum(r["agrees"] for r in comparisons),
        "rows": comparisons,
        "note": "手写开发复核样例的一致性，不是独立人类金标准或用户学习收益。",
    }
    save_json(output / "calibration.json", result)
    return result
