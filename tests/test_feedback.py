import json

from fastapi.testclient import TestClient
from test_learning import Scripted, call, setup

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.provider import ModelError
from learning_agent.store import Store


def note(topic="归一化", confusion="把位置与残差直通混淆", approach="step_by_step"):
    return {
        "topic": topic,
        "confusion": confusion,
        "approach": approach,
        "reason": "围绕用户当前卡点解释，不推出掌握情况",
    }


def deliver(answer, **changes):
    return call("deliver_explanation", {**note(), **changes, "answer": answer})


def test_feedback_anchors_actual_answer_and_state_survives_restart(tmp_path):
    client, model, _, _, conv = setup(
        tmp_path,
        [
            deliver("pre=x+F(N(x))，归一化不包住跳连。"),
            deliver("设F输出0，输入[1,3]。", approach="example"),
        ],
    )
    base = f"/api/conversations/{conv['id']}"
    first = client.post(base + "/messages", json={"content": "解释残差通路"}).json()
    response = client.post(
        base + "/messages",
        json={
            "content": "请换个例子",
            "feedback": {
                "kind": "example",
                "target_run_id": first["run_id"],
                "focus_text": "F输出0",
            },
        },
    )
    assert response.status_code == 200
    history = model.inputs[1][-1]["content"]
    assert "pre=x+F(N(x))" in history and first["run_id"] in history and "example" in history
    state = client.get(base + "/learning").json()
    assert state["working_note"]["approach"] == "example"
    assert [a["approach"] for a in state["attempts"]] == ["step_by_step", "example"]
    assert state["feedback"][0]["target_run_id"] == first["run_id"]
    # Delivery is terminal: no additional model request to store notes or restate the answer.
    assert len(model.inputs) == 2
    with TestClient(
        create_app(Settings(_env_file=None, data_root=tmp_path), Scripted([]))
    ) as restarted:
        assert restarted.get(base + "/learning").json() == state


def test_foreign_feedback_and_forged_mastery_are_rejected_without_new_turn(tmp_path):
    client, model, space, _, conv = setup(tmp_path, [{"role": "assistant", "content": "讲解"}])
    base = f"/api/conversations/{conv['id']}"
    first = client.post(base + "/messages", json={"content": "解释"}).json()
    other = client.post(f"/api/spaces/{space['id']}/conversations").json()["id"]
    assert (
        client.post(
            f"/api/conversations/{other}/messages",
            json={
                "content": "不懂",
                "feedback": {"kind": "stuck", "target_run_id": first["run_id"]},
            },
        ).status_code
        == 422
    )
    assert client.get(f"/api/conversations/{other}/messages").json() == []
    assert (
        client.post(
            base + "/messages",
            json={
                "content": "懂了",
                "feedback": {
                    "kind": "understood",
                    "target_run_id": first["run_id"],
                    "mastery": True,
                },
            },
        ).status_code
        == 422
    )
    model.responses = [
        call("deliver_explanation", {**note(), "answer": "讲解", "mastered": True}),
        {"role": "assistant", "content": "继续解释"},
    ]
    result = client.post(base + "/messages", json={"content": "不懂"}).json()
    events = client.get("/api/runs/" + result["run_id"]).json()["events"]
    assert events[1]["observation"]["error"] == "invalid_arguments"
    assert client.get(base + "/learning").json()["working_note"] is None


def test_failed_response_preserves_feedback_and_prior_working_note(tmp_path):
    client, model, _, _, conv = setup(tmp_path, [deliver("原回答")])
    base = f"/api/conversations/{conv['id']}"
    first = client.post(base + "/messages", json={"content": "问题"}).json()
    prior = client.get(base + "/learning").json()["working_note"]
    model.responses = [
        call("deliver_explanation", {**note(approach="prerequisite"), "answer": ""}),
        ModelError("offline"),
    ]
    failed = client.post(
        base + "/messages",
        json={
            "content": "这一步不懂",
            "feedback": {"kind": "stuck", "target_run_id": first["run_id"]},
        },
    ).json()
    assert failed["state"] == "failed"
    state = client.get(base + "/learning").json()
    assert state["working_note"] == prior
    assert state["feedback"][0]["response_state"] == "failed"
    assert (
        client.post(
            base + "/messages",
            json={
                "content": "懂了",
                "feedback": {"kind": "understood", "target_run_id": failed["run_id"]},
            },
        ).status_code
        == 422
    )


def test_old_answer_feedback_does_not_automatically_replace_new_topic_or_assess_mastery(tmp_path):
    client, model, _, _, conv = setup(tmp_path, [deliver("归一化解释")])
    base = f"/api/conversations/{conv['id']}"
    first = client.post(base + "/messages", json={"content": "讲归一化"}).json()
    model.responses = [
        deliver(
            "换到RoPE", topic="旋转位置编码", confusion="相对位置如何进入内积", approach="explain"
        )
    ]
    client.post(base + "/messages", json={"content": "换个主题讲RoPE"})
    model.responses = [
        deliver(
            "收到，你说的是之前归一化那一点。",
            topic="确认归一化反馈",
            confusion="旧主题用户反馈已明白",
            approach="acknowledge",
        )
    ]
    client.post(
        base + "/messages",
        json={
            "content": "之前那一点懂了",
            "feedback": {"kind": "understood", "target_run_id": first["run_id"]},
        },
    )
    state = client.get(base + "/learning").json()
    assert state["working_note"]["topic"] == "旋转位置编码"
    assert state["feedback"][0]["kind"] == "understood"
    assert "mastered" not in json.dumps(state)
    assert "归一化解释" in model.inputs[-1][-1]["content"]


def test_direct_text_fallback_does_not_fabricate_working_notes(tmp_path):
    client, _, _, _, conv = setup(tmp_path, [{"role": "assistant", "content": "可以直接解释。"}])
    base = f"/api/conversations/{conv['id']}"
    assert (
        client.post(base + "/messages", json={"content": "通用解释"}).json()["state"] == "completed"
    )
    state = Store(tmp_path).learning_state(conv["id"])
    assert state["attempts"] == [] and state["feedback"] == []


def test_delivery_must_wait_for_other_tool_observations(tmp_path):
    premature = deliver("已根据实际目录交付")
    premature["tool_calls"].extend(call("list_materials", {}, "list-call")["tool_calls"])
    client, model, _, _, conv = setup(tmp_path, [premature, deliver("已根据实际目录交付")])
    base = f"/api/conversations/{conv['id']}"
    result = client.post(base + "/messages", json={"content": "查阅目录解释"}).json()
    events = client.get("/api/runs/" + result["run_id"]).json()["events"]
    assert events[1]["observation"]["error"] == "deliver_after_observations"
    assert result["answer"] == "已根据实际目录交付" and len(model.inputs) == 2
    assert len(client.get(base + "/learning").json()["attempts"]) == 1
