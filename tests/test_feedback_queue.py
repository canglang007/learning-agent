import json

import pytest

from learning_agent.evaluation import queue as queue_store


def candidate(case_id="fb-1", concepts=None):
    turns = [
        {
            "id": "explain",
            "conversation": "main",
            "input": {"question": "为什么近似相等要写误差"},
            "gold": {"required_concepts": [], "evidence_targets": []},
            "actions": [],
        },
        {
            "id": "followup",
            "conversation": "main",
            "input": {
                "question": "这一步没懂，误差为什么要除以真值",
                "feedback": {
                    "kind": "stuck",
                    "focus_text": "分母为什么取真值",
                    "target_turn": "explain",
                },
            },
            "gold": {"required_concepts": list(concepts or []), "evidence_targets": []},
            "actions": [],
        },
    ]
    return {
        "run_id": case_id,
        "conversation_id": "conversation",
        "score": 4,
        "signals": [{"code": "feedback_stuck", "weight": 4, "detail": "用户说没懂"}],
        "question": "这一步没懂，误差为什么要除以真值",
        "target_question": "为什么近似相等要写误差",
        "feedback": {"kind": "stuck", "focus_text": "分母为什么取真值"},
        "duplicate_of": None,
        "suggested_case": {
            "id": case_id,
            "family": "feedback",
            "split": "dev",
            "tags": ["from-usage"],
            "turns": turns,
        },
    }


def existing_suite():
    return {
        "id": "learning-core",
        "version": "1",
        "rubric_version": "1",
        "cases": [
            {
                "id": "old_case",
                "family": "old",
                "split": "dev",
                "turns": [
                    {
                        "id": "t",
                        "input": {"question": "已有问题"},
                        "gold": {"required_concepts": ["x"]},
                    }
                ],
            }
        ],
    }


def test_merge_adds_pending_and_updates_only_pending(tmp_path):
    data = queue_store.load(tmp_path / "queue.json")
    first = queue_store.merge(data, [candidate()])
    assert first == {"added": 1, "updated": 0, "keeping_reviewed": 0}
    assert data["candidates"]["fb-1"]["status"] == "pending"

    queue_store.review(data, "fb-1", "accepted", concepts=["要说明误差范围"])
    again = queue_store.merge(data, [candidate()])
    assert again == {"added": 0, "updated": 0, "keeping_reviewed": 1}
    entry = data["candidates"]["fb-1"]
    assert entry["status"] == "accepted"
    # 重跑挖掘不能把人工补的gold冲掉
    assert entry["case"]["turns"][-1]["gold"]["required_concepts"] == ["要说明误差范围"]


def test_review_writes_gold_into_the_turn_being_tested(tmp_path):
    data = queue_store.load(tmp_path / "queue.json")
    queue_store.merge(data, [candidate()])
    entry = queue_store.review(
        data,
        "fb-1",
        "accepted",
        note="确认是真卡点",
        concepts=["相对误差分母取真值"],
        forbidden=["任何误差都算近似相等"],
    )
    gold = entry["case"]["turns"][-1]["gold"]
    assert gold["required_concepts"] == ["相对误差分母取真值"]
    assert gold["forbidden_claims"] == ["任何误差都算近似相等"]
    assert entry["note"] == "确认是真卡点"
    assert entry["reviewed_at"]
    assert queue_store.gold_ready(entry["case"]) is True


def test_review_rejects_unknown_candidate(tmp_path):
    data = queue_store.load(tmp_path / "queue.json")
    with pytest.raises(LookupError):
        queue_store.review(data, "fb-missing", "accepted")
    queue_store.merge(data, [candidate()])
    with pytest.raises(ValueError):
        queue_store.review(data, "fb-1", "maybe")


def test_promote_only_takes_accepted_with_gold_and_bumps_version(tmp_path):
    data = queue_store.load(tmp_path / "queue.json")
    queue_store.merge(data, [candidate("fb-a"), candidate("fb-b")])
    queue_store.review(data, "fb-a", "accepted", concepts=["要求1"])
    # 接受但gold还空着：应当被挡住，避免promote出只有复核项的case
    queue_store.review(data, "fb-b", "accepted")

    payload, added, blocked = queue_store.promote(data, existing_suite(), version="2")
    assert added == ["fb-a"]
    assert blocked == ["fb-b"]
    assert payload["version"] == "2"
    assert [case["id"] for case in payload["cases"]] == ["old_case", "fb-a"]


def test_promote_is_idempotent(tmp_path):
    data = queue_store.load(tmp_path / "queue.json")
    queue_store.merge(data, [candidate()])
    queue_store.review(data, "fb-1", "accepted", concepts=["要求"])
    first, added, _ = queue_store.promote(data, existing_suite(), version="2")
    assert added == ["fb-1"]
    second, again, _ = queue_store.promote(data, first, version="3")
    assert again == []
    assert [case["id"] for case in second["cases"]] == ["old_case", "fb-1"]


def test_queue_roundtrip_and_version_guard(tmp_path):
    path = tmp_path / "nested" / "queue.json"
    data = queue_store.load(path)
    queue_store.merge(data, [candidate()])
    queue_store.save(path, data)
    assert path.is_file()
    again = queue_store.load(path)
    assert again["candidates"]["fb-1"]["status"] == "pending"
    assert again["updated_at"]

    path.write_text(json.dumps({"schema_version": 99, "candidates": {}}))
    with pytest.raises(ValueError):
        queue_store.load(path)
