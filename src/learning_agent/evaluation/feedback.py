"""从真实使用记录里挑回归候选：只排序和给证据，不自动写gold。"""

import json
import re
from pathlib import Path

from learning_agent.evaluation.collector import collect, read_store
from learning_agent.evaluation.graders import grade
from learning_agent.evaluation.schemas import Suite

# 反馈类型先验：用户明确说没懂，比"想换个例子"更值得回流
FEEDBACK_WEIGHTS = {
    "stuck": (4, "用户点了这一步没懂：当前讲法没有解决这个卡点"),
    "restate": (3, "用户要求检查自己的复述：复述是否成立需要复核"),
    "example": (2, "用户要求换例子：抽象讲法不够"),
    "understood": (-2, "用户自述明白了：对照正例，价值低于失败样本"),
}

STATE_WEIGHTS = {
    "failed": (4, "运行失败"),
    "interrupted": (2, "运行中断，可能是环境问题或恢复实验，需人工判断"),
    "budget_exhausted": (2, "预算耗尽，答案可能不完整"),
    "cancelled": (1, "用户取消"),
}

INCOMPLETE = {"queued", "running", "cancelling"}
DELIVERED = {"completed", "budget_exhausted"}
REFUSAL_WORDING = (
    "没有找到",
    "未覆盖",
    "资料里没有",
    "无法确认",
    "没有提到",
    "未提及",
    "不包含",
    "没有相关",
)

CAVEAT = (
    "分数只用于排序，不是缺陷判定；证据位置来自模型实际观察，不等于答案所在；"
    "required_concepts与forbidden_claims必须人工补，空着跑出来的只有复核项。"
)


def normalize(text):
    """比较问题是否等价时去掉空白和标点，保留中英文字符。"""
    return re.sub(r"[\s\W_]+", "", (text or "").lower())


def known_questions(eval_dir):
    """已进评测集的问题文本；回流前先去重，避免同一问题反复入库。"""
    known = {}
    if not eval_dir:
        return known
    directory = Path(eval_dir)
    if not directory.is_dir():
        return known
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for case in data.get("cases", []) or []:
            for turn in case.get("turns", []) or []:
                question = (turn.get("input") or {}).get("question")
                if question:
                    known.setdefault(normalize(question), f"{case.get('id')}/{turn.get('id')}")
    return known


def material_of(entry):
    """证据条目可能只带asset_path，从路径里取资料ID。"""
    if entry.get("material_id"):
        return entry["material_id"]
    parts = (entry.get("asset_path") or "").split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "materials" else None


def locations(trace, names, limit=3):
    """模型在这次运行里真正看过的位置，按出现顺序取前几个。"""
    found, seen = [], set()
    for operation in trace.get("operations", []):
        ref = operation.get("input_ref") or {}
        if not isinstance(ref, dict) or ref.get("schema_version") != 1:
            continue
        for kind in ("images", "evidence"):
            for entry in ref.get(kind) or []:
                if not isinstance(entry, dict):
                    continue
                material_id = material_of(entry)
                # 只有文本没有位置的观察（例如交付工具）不构成证据位置
                if not material_id and not entry.get("page") and entry.get("seconds") is None:
                    continue
                key = (material_id, entry.get("page"), entry.get("seconds"))
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "material_id": material_id,
                        "name": names.get(material_id),
                        "page": entry.get("page"),
                        "start_seconds": entry.get("start_seconds"),
                        "end_seconds": entry.get("end_seconds"),
                        "seconds": entry.get("seconds"),
                        "image": kind == "images",
                    }
                )
                if len(found) >= limit:
                    return found
    return found


def targets(found):
    """把观察到的位置转成evidence_targets草稿；页码和时间点都保留原始口径。"""
    result, seen = [], set()
    for location in found:
        name = location.get("name") or location.get("material_id")
        if not name:
            continue
        if location.get("page"):
            entry = {
                "asset": name,
                "pages": [location["page"]],
                "modality": "image" if location.get("image") else "any",
            }
        elif location.get("start_seconds") is not None and location.get("end_seconds") is not None:
            middle = (location["start_seconds"] + location["end_seconds"]) / 2
            entry = {"asset": name, "anchors": [round(middle, 1)]}
        else:
            continue
        key = json.dumps(entry, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


def assess(trace, target, feedback, names):
    """一条运行值不值得回流；没有用户问题就跳过，不猜。"""
    messages = trace.get("messages", [])
    question = next((m["content"] for m in messages if m.get("role") == "user"), None)
    if not question:
        return None
    answer = trace.get("answer")
    signals = []

    def note(code, weight, detail):
        signals.append({"code": code, "weight": weight, "detail": detail})

    if feedback:
        weight, detail = FEEDBACK_WEIGHTS.get(feedback["kind"], (1, "未知反馈类型"))
        note("feedback_" + feedback["kind"], weight, detail)
    if trace.get("state") in STATE_WEIGHTS:
        note("state_" + trace["state"], *STATE_WEIGHTS[trace["state"]])
    if trace.get("state") in DELIVERED and not answer:
        note("missing_canonical_answer", 3, "运行已完成但没有canonical答案")
    tools = trace.get("tools", [])
    if answer and len(answer) > 200 and not tools:
        note("answer_without_observation", 2, "没有任何工具观察却给出长答案，需确认不是凭空作答")
    errors = [t for t in tools if (t.get("observation") or {}).get("error")]
    if errors:
        note("tool_error", 3, "工具观察含错误：" + str(errors[0]["observation"]["error"])[:80])
    if answer and any(word in answer for word in REFUSAL_WORDING):
        note("refusal_wording", 1, "回答含拒答措辞，需确认资料确实不覆盖")

    delivery = next(
        (c["status"] for c in grade(trace, {}, {})["checks"] if c["name"] == "runtime_delivery"),
        None,
    )
    if delivery == "fail" and not any(s["code"] == "missing_canonical_answer" for s in signals):
        note("delivery_failed", 3, "canonical答案交付失败，与grader的runtime_delivery同口径")

    own = locations(trace, names)
    seen = locations(target, names) if target else []
    target_question = None
    if target:
        target_question = next(
            (m["content"] for m in target.get("messages", []) if m.get("role") == "user"),
            None,
        )
    return {
        "question": question,
        "target_question": target_question,
        "answer_excerpt": (answer or "")[:400],
        "delivery": delivery,
        "feedback": (
            {"kind": feedback["kind"], "focus_text": feedback.get("focus_text") or ""}
            if feedback
            else None
        ),
        "signals": signals,
        "evidence_targets": targets(own),
        "target_evidence": targets(seen),
        "observed_locations": own,
        "target_observed_locations": seen,
    }


def build_case(candidate):
    """反馈类候选天生是两轮：原问题一轮，反馈轮挂在它后面。"""
    case_id = "fb-" + re.sub(r"[^a-z0-9]", "", candidate["run_id"].lower())[:10]
    explain = {
        "id": "explain",
        "conversation": "main",
        "input": {"question": candidate["target_question"] or candidate["question"]},
        "gold": {
            "required_concepts": [],
            "evidence_targets": candidate.get("target_evidence") or candidate["evidence_targets"],
        },
        "actions": [],
    }
    feedback = candidate.get("feedback")
    turns = [explain]
    if feedback:
        turns.append(
            {
                "id": "followup",
                "conversation": "main",
                "input": {
                    "question": candidate["question"],
                    "feedback": {
                        "kind": feedback["kind"],
                        "focus_text": feedback.get("focus_text") or candidate["question"],
                        "target_turn": "explain",
                    },
                },
                "gold": {
                    "required_concepts": [],
                    "evidence_targets": candidate["evidence_targets"],
                },
                "actions": [],
            }
        )
    return {
        "id": case_id,
        "family": "feedback",
        "split": "dev",
        "tags": ["from-usage"],
        "turns": turns,
    }


def build_suite(candidates, suite_id="feedback-candidates"):
    payload = {
        "id": suite_id,
        "version": "1",
        "rubric_version": "1",
        "cases": [candidate["suggested_case"] for candidate in candidates],
    }
    # 提前校验：导出的片段必须能直接喂给run子命令
    Suite.model_validate(payload)
    return payload


def mine(workspace, *, eval_dir=None, since=None, limit=40, min_score=1):
    """只读扫描运行与反馈；不调用模型，不写数据库。"""
    store = read_store(workspace)
    known = known_questions(eval_dir)
    with store.connect() as db:
        rows = [
            dict(row)
            for row in db.execute(
                "SELECT id,conversation_id,state,created_at FROM runs ORDER BY rowid"
            )
        ]
        feedback = {
            row["run_id"]: dict(row)
            for row in db.execute(
                "SELECT run_id,target_run_id,kind,focus_text FROM learning_feedback"
            )
        }
        names = {row["id"]: row["name"] for row in db.execute("SELECT id,name FROM materials")}

    traces = {}

    def trace_of(run_id):
        if run_id not in traces:
            try:
                traces[run_id] = collect(store, run_id)
            except (LookupError, FileNotFoundError):
                traces[run_id] = None
        return traces[run_id]

    skipped = {"incomplete": 0, "before_since": 0, "no_question": 0}
    candidates = []
    for row in rows:
        run_id = row["id"]
        if row["state"] in INCOMPLETE:
            skipped["incomplete"] += 1
            continue
        if since and (row.get("created_at") or "")[:10] < since:
            skipped["before_since"] += 1
            continue
        trace = trace_of(run_id)
        if trace is None:
            skipped["no_question"] += 1
            continue
        item = feedback.get(run_id)
        target = trace_of(item["target_run_id"]) if item else None
        record = assess(trace, target, item, names)
        if record is None:
            skipped["no_question"] += 1
            continue
        record.update(
            run_id=run_id,
            conversation_id=row["conversation_id"],
            state=row["state"],
            created_at=row.get("created_at"),
            duplicate_of=known.get(normalize(record["target_question"] or record["question"])),
            needs_human_gold=True,
        )
        if record["duplicate_of"]:
            record["signals"].append(
                {"code": "already_in_eval", "weight": -5, "detail": "同样的问题已在评测集"}
            )
        record["score"] = sum(signal["weight"] for signal in record["signals"])
        record["suggested_case"] = build_case(record)
        candidates.append(record)

    kept = [item for item in candidates if item["score"] >= min_score]
    kept.sort(key=lambda item: (-item["score"], item.get("created_at") or ""))
    return {
        "kind": "feedback_candidates",
        "schema_version": 1,
        "workspace": str(Path(workspace).resolve()),
        "scanned": len(rows),
        "skipped": skipped,
        "eval_questions": len(known),
        "candidates_found": len(candidates),
        "candidates": kept[:limit],
        "caveat": CAVEAT,
    }
