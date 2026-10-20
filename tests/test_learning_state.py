import asyncio
import json

import pytest
from client_helpers import TestClient, close_for
from test_feedback import deliver
from test_learning import call, setup

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import ContextManager, PreparedContext, ProtocolGroup, TurnInput
from learning_agent.provider import ModelError
from learning_agent.store import Store
from learning_agent.summaries import maybe_summarize


def test_state_corrections_revision_conflict_and_reset_survive_restart(tmp_path):
    client, model, _, _, conv = setup(
        tmp_path,
        [deliver("讲解", state_update={"goal": "理解归一化", "add_questions": ["为什么有残差？"]})],
    )
    base = f"/api/conversations/{conv['id']}"
    client.post(base + "/messages", json={"content": "讲归一化"})
    original = client.get(base + "/learning").json()
    assert original["state"]["goal"]["origin"] == "agent_hypothesis"
    changed = client.patch(
        base + "/learning",
        json={
            "revision": original["revision"],
            "goal": "我想推导梯度",
            "confusion": "卡在链式法则",
        },
    ).json()
    assert changed["state"]["goal"]["origin"] == "user_explicit"
    assert (
        client.patch(
            base + "/learning", json={"revision": original["revision"], "goal": "旧版本"}
        ).status_code
        == 409
    )
    model.responses = [
        deliver(
            "继续",
            confusion="错误的新卡点",
            state_update={"goal": "模型猜测", "user_quote": "继续"},
        )
    ]
    client.post(base + "/messages", json={"content": "继续"})
    current = client.get(base + "/learning").json()
    assert current["state"]["goal"]["text"] == "我想推导梯度"
    assert current["state"]["confusion"]["text"] == "卡在链式法则"
    reset = client.patch(
        base + "/learning", json={"revision": current["revision"], "reset": True}
    ).json()
    assert reset["state"]["goal"] is None and reset["state"]["open_questions"] == []
    close_for(tmp_path)
    reopened = TestClient(create_app(Settings(_env_file=None, data_root=tmp_path), model))
    assert reopened.get(base + "/learning").json()["state"] == reset["state"]
    assert len(reopened.get(base + "/messages").json()) == 4


def test_resolution_requires_matching_user_feedback_and_old_topic_is_protected(tmp_path):
    client, model, _, _, conv = setup(
        tmp_path, [deliver("解释A", state_update={"add_questions": ["A的前提是什么？"]})]
    )
    base = f"/api/conversations/{conv['id']}"
    first = client.post(base + "/messages", json={"content": "解释A"}).json()
    old = client.get(base + "/learning").json()
    qid = old["state"]["open_questions"][0]["id"]
    model.responses = [
        deliver("无依据关闭", state_update={"resolve_question_ids": [qid]}),
        {"role": "assistant", "content": "仍需确认。"},
    ]
    client.post(base + "/messages", json={"content": "继续"})
    assert client.get(base + "/learning").json()["state"] == old["state"]
    model.responses = [deliver("讲B", topic="B", confusion="B卡点")]
    client.post(base + "/messages", json={"content": "换到B"})
    model.responses = [
        deliver(
            "确认A",
            topic="A",
            approach="acknowledge",
            state_update={
                "active_topic": "A",
                "confusion": "A已理解",
                "resolve_question_ids": [qid],
            },
        )
    ]
    result = client.post(
        base + "/messages",
        json={
            "content": "A明白了",
            "feedback": {"kind": "understood", "target_run_id": first["run_id"]},
        },
    ).json()
    assert result["state"] == "completed"
    state = client.get(base + "/learning").json()["state"]
    assert state["active_topic"]["text"] == "B" and state["confusion"]["text"] == "B卡点"
    assert state["open_questions"][0]["status"] == "user_understood"


def test_pinned_goal_survives_literal_topic_switch_and_user_can_unpin(tmp_path):
    client, model, _, _, conv = setup(tmp_path, [deliver("起始解释")])
    base = f"/api/conversations/{conv['id']}"
    client.post(base + "/messages", json={"content": "开始"})
    revision = client.get(base + "/learning").json()["revision"]
    changed = client.patch(
        base + "/learning", json={"revision": revision, "goal": "用验证证据选择正则化"}
    ).json()
    assert changed["state"]["goal_pinned"]
    patch = {"goal": "梯度下降", "active_topic": "梯度下降", "user_quote": "换个主题讲梯度下降"}
    model.responses = [deliver("讲梯度", topic="梯度下降", state_update=patch)]
    client.post(base + "/messages", json={"content": "换个主题讲梯度下降"})
    result = client.get(base + "/learning").json()
    assert result["state"]["goal"]["text"] == "用验证证据选择正则化"
    assert result["state"]["active_topic"]["text"] == "梯度下降"
    client.patch(base + "/learning", json={"revision": result["revision"], "goal_pinned": False})
    model.responses = [deliver("讲梯度", topic="梯度下降", state_update=patch)]
    client.post(base + "/messages", json={"content": "换个主题讲梯度下降"})
    assert client.get(base + "/learning").json()["state"]["goal"]["text"] == "梯度下降"


def test_history_reader_is_scoped_bounded_and_available_to_agent(tmp_path):
    client, model, space, _, conv = setup(tmp_path, [{"role": "assistant", "content": "原回答"}])
    base = f"/api/conversations/{conv['id']}"
    first = client.post(base + "/messages", json={"content": "原问题的具体说法"}).json()
    other = client.post(f"/api/spaces/{space['id']}/conversations").json()["id"]
    model.responses = [
        call(
            "recall_learning_context",
            {"run_id": first["run_id"], "role": "user", "start_char": 2, "max_chars": 3},
        ),
        {"role": "assistant", "content": "已核对原话。"},
    ]
    result = client.post(base + "/messages", json={"content": "核对原话"}).json()
    assert result["state"] == "completed"
    observation = json.loads(model.inputs[-1][-1]["content"])
    assert observation["content"] == "题的具" and observation["truncated"]
    model.responses = [
        call("recall_learning_context", {"run_id": first["run_id"]}),
        {"role": "assistant", "content": "不能读取其他会话。"},
    ]
    client.post(f"/api/conversations/{other}/messages", json={"content": "读取别的会话"})
    assert json.loads(model.inputs[-1][-1]["content"])["error"] == "historical_context_not_found"
    assert (
        Store(tmp_path).recall_learning_context(conv["id"], None, "assistant", 0, 100, 1)["content"]
        == "原回答"
    )


class Summarizer:
    def __init__(self, bad=False):
        self.calls = 0
        self.bad = bad

    async def step(self, history, instructions, tools, allow):
        self.calls += 1
        source = json.loads(history[0]["content"])["sources"][0]
        return call(
            "save_context_summary",
            {
                "items": [
                    {
                        "kind": "user_statement",
                        "text": "用户希望保留最初目标",
                        "source_run_ids": ["foreign-id"] if self.bad else [source["run_id"]],
                        "user_quote": source["question"],
                    }
                ]
            },
        ), {"prompt_tokens": 100, "completion_tokens": 20}


def archived_turns(tmp_path, answer="原始解释"):
    store = Store(tmp_path)
    conv = store.conversation(store.space("学习")["id"])["id"]
    settings = Settings(_env_file=None, data_root=tmp_path)
    manager = ContextManager(settings)
    store.context_turns(conv)
    for n in range(8):
        question = f"第{n + 1}个原始问题"
        run = store.start_run(conv, question)
        store.finish_run(
            run,
            "completed",
            answer,
            [],
            archive={
                "profile": manager.profile,
                "input": {"question": question},
                "groups": [
                    ProtocolGroup(
                        f"{run}:input", "input", [{"role": "user", "content": question}]
                    ).dump(),
                    ProtocolGroup(
                        f"{run}:answer", "answer", [{"role": "assistant", "content": answer}]
                    ).dump(),
                ],
            },
        )
    return store, conv, settings, manager


def test_summary_sources_watermark_reuse_and_user_correction_invalidation(tmp_path):
    store, conv, settings, manager = archived_turns(tmp_path)
    turns = store.context_turns(conv)
    state = store.current_learning_state(conv)
    prepared = PreparedContext([], {"omissions": [{"reason": "history_budget_pressure"}]}, set())
    provider = Summarizer()
    summary, event = asyncio.run(
        maybe_summarize(
            store, provider, settings, conv, turns, state, manager.profile, prepared, None
        )
    )
    assert summary["status"] == "ready" and event["usage"]["prompt_tokens"] == 100
    assert summary["covered_end"] == 5
    reused, event = asyncio.run(
        maybe_summarize(
            store, provider, settings, conv, turns, state, manager.profile, prepared, summary
        )
    )
    assert reused["id"] == summary["id"] and event is None and provider.calls == 1
    selected = manager.prepare(
        turns,
        [ProtocolGroup("new", "input", [{"role": "user", "content": "继续"}])],
        TurnInput("继续"),
        "",
        [],
        True,
        summary,
    )
    assert selected.snapshot["summary_id"] == summary["id"]
    assert any("source_manifest" in m["content"] for m in selected.messages)
    assert not any(m.get("content") == "第1个原始问题" for m in selected.messages)
    store.edit_learning_state(conv, {"revision": state["revision"], "goal": "新目标"})
    assert (
        store.latest_summary_attempt(conv, state["context_epoch"], manager.profile, ready=True)
        is None
    )


def test_invalid_summary_does_not_replace_old_state_or_retry_same_range(tmp_path):
    store, conv, settings, manager = archived_turns(tmp_path)
    turns = store.context_turns(conv)
    state = store.current_learning_state(conv)
    prepared = PreparedContext([], {"omissions": [{"reason": "tool_budget_pressure"}]}, set())
    provider = Summarizer(bad=True)
    summary, event = asyncio.run(
        maybe_summarize(
            store, provider, settings, conv, turns, state, manager.profile, prepared, None
        )
    )
    assert (
        summary is None and event["state"] == "failed" and event["usage"]["completion_tokens"] == 20
    )
    assert store.current_learning_state(conv) == state
    again, event = asyncio.run(
        maybe_summarize(
            store, provider, settings, conv, turns, state, manager.profile, prepared, None
        )
    )
    assert again is None and event is None and provider.calls == 1


def test_edit_rejects_running_turn_and_evidence_revision_changes(tmp_path):
    client, model, space, material, conv = setup(tmp_path, [])
    model.responses = [
        call("read_passage", {"material_id": material["id"], "start_line": 1, "end_line": 2}),
        deliver("已核对"),
    ]
    base = f"/api/conversations/{conv['id']}"
    client.post(base + "/messages", json={"content": "解释原文"})
    state = client.get(base + "/learning").json()
    assert state["state"]["evidence_refs"][0]["version_status"] == "current_version"
    store = Store(tmp_path)
    store.set_source(space["id"], material["id"], {"url": "https://example.com/source"})
    assert (
        client.get(base + "/learning").json()["state"]["evidence_refs"][0]["version_status"]
        == "stale_or_missing"
    )
    store.start_run(conv["id"], "尚未完成")
    assert (
        client.patch(
            base + "/learning", json={"revision": state["revision"], "goal": "并发修改"}
        ).status_code
        == 409
    )


@pytest.mark.parametrize("failure", ["timeout", "service_error"])
def test_summary_failure_falls_back_to_regular_answer_without_same_range_retry(tmp_path, failure):
    store, conv, settings, _ = archived_turns(tmp_path, answer="原始解释" * 2000)

    class Teacher:
        summary_calls = 0

        async def step(self, history, instructions, tools, allow):
            if tools[0]["name"] == "save_context_summary":
                self.summary_calls += 1
                if failure == "timeout":
                    await asyncio.sleep(0.02)
                raise ModelError("unavailable")
            return deliver("摘要暂不可用，仍可讲解当前问题。"), {"total_tokens": 10}

    model = Teacher()
    settings = Settings(
        _env_file=None,
        **{
            **settings.model_dump(),
            "context_budget_tokens": 24000,
            "context_output_reserve_tokens": 512,
            "context_headroom_tokens": 512,
            "summary_timeout_seconds": 0.001,
        },
    )
    close_for(tmp_path)
    client = TestClient(create_app(settings, model))
    for _ in range(2):
        result = client.post(
            f"/api/conversations/{conv}/messages", json={"content": "当前问题"}
        ).json()
        assert result["state"] == "completed"
    assert model.summary_calls == 1
    with store.connect() as db:
        assert db.execute("SELECT status FROM context_summaries").fetchone()[0] == "failed"
