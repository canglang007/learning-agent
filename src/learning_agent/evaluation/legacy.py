import json
from pathlib import Path

from learning_agent.evaluation.ledger import save_json


def import_legacy(source, output):
    data = json.loads(Path(source).read_text())
    rows = []
    if isinstance(data, list):
        kind = "course_retrieval"
        rows = [
            {
                "mode": r.get("mode"),
                "annotated_hit_at_5": r.get("hit_at_5"),
                "annotated_mrr_at_5": r.get("mrr_at_5"),
                "query_count": len(r.get("rows", [])),
            }
            for r in data
        ]
    else:
        kind = "real_courses" if "cases" in data else "runtime_rt3"
        turns = (
            [t for c in data.get("cases", []) for t in c.get("turns", [])]
            if kind == "real_courses"
            else data.get("turns", [])
        )
        rows = [
            {
                "run_id": t.get("run_id"),
                "state": t.get("state"),
                "latency_s": t.get("latency_s", t.get("total_s")),
                "review": None,
                "input_manifest": "unknown",
                "raw_review_present": bool(t.get("review")),
            }
            for t in turns
        ]
    result = {
        "schema_version": 1,
        "kind": "legacy:" + kind,
        "source": Path(source).name,
        "rows": rows,
        "note": "原报告缺少的attempt输入/split/独立复核均未知；不转写为新模块教学通过成绩。",
    }
    save_json(output, result)
    return result
