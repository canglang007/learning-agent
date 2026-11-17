"""候选审核队列：本机状态，含问题原文，不进仓库。"""

import json
from datetime import UTC, datetime
from pathlib import Path

from learning_agent.evaluation.schemas import Suite

QUEUE_VERSION = 1
STATUSES = ("pending", "accepted", "rejected")
DEFAULT_QUEUE = Path("data/feedback/queue.json")


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def load(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": QUEUE_VERSION, "updated_at": None, "candidates": {}}
    data = json.loads(path.read_text())
    if data.get("schema_version") != QUEUE_VERSION:
        raise ValueError(f"队列版本不支持：{data.get('schema_version')}")
    data.setdefault("candidates", {})
    return data


def save(path, queue):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    queue["schema_version"] = QUEUE_VERSION
    queue["updated_at"] = now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2))
    temporary.replace(path)


def merge(queue, candidates):
    """新候选入库；已审过的不动，重跑挖掘不该把 accepted 打回 pending。"""
    counts = {"added": 0, "updated": 0, "keeping_reviewed": 0}
    for item in candidates:
        case = item["suggested_case"]
        key = case["id"]
        machine = {
            "case_id": key,
            "run_id": item["run_id"],
            "conversation_id": item.get("conversation_id"),
            "score": item["score"],
            "signals": item["signals"],
            "question": item["target_question"] or item["question"],
            "feedback": item.get("feedback"),
            "duplicate_of": item.get("duplicate_of"),
            "case": case,
            "needs_human_gold": True,
        }
        entry = queue["candidates"].get(key)
        if entry is None:
            queue["candidates"][key] = {
                **machine,
                "status": "pending",
                "note": "",
                "created_at": now(),
                "reviewed_at": None,
            }
            counts["added"] += 1
        elif entry.get("status") == "pending":
            entry.update(machine)
            counts["updated"] += 1
        else:
            counts["keeping_reviewed"] += 1
    return counts


def review(queue, case_id, status, note="", concepts=None, forbidden=None):
    """定状态，同时把人工补的gold写进要验的那一轮。"""
    entry = queue["candidates"].get(case_id)
    if entry is None:
        raise LookupError(f"候选不存在：{case_id}")
    if status not in STATUSES:
        raise ValueError(f"未知状态：{status}，可选 {'/'.join(STATUSES)}")
    turns = (entry.get("case") or {}).get("turns") or []
    if status == "accepted" and not turns:
        raise ValueError("候选缺少case结构，无法接受")
    if turns and (concepts or forbidden):
        gold = turns[-1].setdefault("gold", {})
        for concept in concepts or []:
            if concept not in gold.setdefault("required_concepts", []):
                gold["required_concepts"].append(concept)
        for claim in forbidden or []:
            if claim not in gold.setdefault("forbidden_claims", []):
                gold["forbidden_claims"].append(claim)
    entry["status"] = status
    entry["note"] = note
    entry["reviewed_at"] = now()
    return entry


def gold_ready(case):
    """空gold的case跑起来只有复核项，promote时挡掉。"""
    for turn in case.get("turns") or []:
        gold = turn.get("gold") or {}
        if gold.get("required_concepts") or gold.get("evidence_targets"):
            return True
        if gold.get("forbidden_claims") or gold.get("numerical_checks"):
            return True
    return False


def promote(queue, suite_payload, *, version=None, statuses=("accepted",)):
    """把已接受的候选并进suite；返回新payload和实际加进去的id。"""
    cases = [dict(case) for case in suite_payload.get("cases") or []]
    existing = {case.get("id") for case in cases}
    added, blocked = [], []
    for entry in queue.get("candidates", {}).values():
        if entry.get("status") not in statuses:
            continue
        case = entry.get("case")
        if not case or case.get("id") in existing:
            continue
        if not gold_ready(case):
            blocked.append(case.get("id"))
            continue
        cases.append(case)
        existing.add(case.get("id"))
        added.append(case.get("id"))
    payload = {**suite_payload, "cases": cases}
    if version is not None:
        payload["version"] = str(version)
    Suite.model_validate(payload)
    return payload, added, blocked
