import copy
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_learning import Scripted, call, setup
from test_pdf import prepared as pdf_setup

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import ContextError, ContextManager, ProtocolGroup, TurnInput
from learning_agent.provider import OpenAIProvider
from learning_agent.store import Store
from learning_agent.tools import MaterialTools


def small_settings(tmp_path, **changes):
    return Settings(
        _env_file=None,
        data_root=tmp_path,
        context_budget_tokens=18000,
        context_output_reserve_tokens=512,
        context_headroom_tokens=512,
        **changes,
    )


def test_long_history_is_omitted_without_deleting_original_or_repeating_injections(tmp_path):
    client, model, _, _, conv = setup(
        tmp_path,
        [
            {"role": "assistant", "content": "old" * 40000},
            {"role": "assistant", "content": "新的讲解"},
        ],
    )
    client = TestClient(create_app(small_settings(tmp_path), model))
    url = f"/api/conversations/{conv['id']}/messages"
    first = client.post(url, json={"content": "旧问题"}).json()
    second = client.post(url, json={"content": "新问题"}).json()
    assert first["state"] == second["state"] == "completed"
    assert second["context_limited"]
    assert "oldold" not in json.dumps(model.inputs[-1])
    assert client.get(url).json()[1]["content"] == "old" * 40000
    contexts = client.get(f"/api/runs/{second['run_id']}/context").json()
    assert contexts[0]["estimate"]["input_total"] <= contexts[0]["input_budget"]
    with sqlite3.connect(tmp_path / "workspace.db") as db:
        archives = db.execute(
            "SELECT raw_input,groups_json FROM protocol_turns WHERE run_id IS NOT NULL"
        ).fetchall()
    assert len(archives) == 2 and "oldold" in archives[0][1]
    assert json.loads(archives[-1][0])["question"] == "新问题"
    assert "会话学习工作记录" not in archives[-1][1]


def test_budget_checked_after_tool_and_failed_groups_archived_but_not_replayed(tmp_path):
    client, model, space, _, conv = setup(tmp_path, [])
    material = client.post(
        f"/api/spaces/{space['id']}/materials",
        files={"file": ("large.txt", ("资料" * 4000).encode())},
    ).json()
    model.responses = [
        call("read_passage", {"material_id": material["id"], "start_line": 1, "end_line": 1}),
        {"role": "assistant", "content": "新的问题可以直接解释。"},
    ]
    client = TestClient(create_app(small_settings(tmp_path), model))
    url = f"/api/conversations/{conv['id']}/messages"
    failed = client.post(url, json={"content": "读取这份资料"}).json()
    assert failed["state"] == "failed" and len(model.inputs) == 1
    diagnostics = client.get(f"/api/runs/{failed['run_id']}/context").json()
    assert len(diagnostics) == 2
    assert diagnostics[-1]["result"]["error"] == "context_required_too_large"
    assert diagnostics[0]["result"]["usage"]["total_tokens"] == 10
    with sqlite3.connect(tmp_path / "workspace.db") as db:
        row = db.execute(
            "SELECT groups_json,replay_status FROM protocol_turns WHERE run_id=?",
            (failed["run_id"],),
        ).fetchone()
    assert row[1] == "failed" and "资料资料" in json.loads(row[0])[-1]["messages"][-1]["content"]
    assert client.post(url, json={"content": "新的问题"}).json()["state"] == "completed"
    assert not any(m["role"] == "tool" for m in model.inputs[-1])


def test_required_question_too_large_never_calls_model(tmp_path):
    client, model, _, _, conv = setup(tmp_path, [])
    client = TestClient(create_app(small_settings(tmp_path), model))
    result = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "大" * 6000}
    ).json()
    assert result["state"] == "failed" and model.inputs == []
    assert (
        client.get(f"/api/runs/{result['run_id']}/context").json()[0]["result"]["error"]
        == "context_required_too_large"
    )


def test_feedback_excerpt_keeps_whole_target_paragraph_and_original_answer(tmp_path):
    answer = "甲" * 6000 + "\n\n关键公式的含义。\n\n" + "乙" * 6000
    client, model, _, _, conv = setup(
        tmp_path,
        [
            {"role": "assistant", "content": answer},
            {"role": "assistant", "content": "针对公式解释。"},
        ],
    )
    client = TestClient(create_app(small_settings(tmp_path), model))
    url = f"/api/conversations/{conv['id']}/messages"
    first = client.post(url, json={"content": "解释"}).json()
    result = client.post(
        url,
        json={
            "content": "这步没懂",
            "feedback": {
                "kind": "stuck",
                "target_run_id": first["run_id"],
                "focus_text": "关键公式的含义",
            },
        },
    ).json()
    assert result["state"] == "completed"
    assert "关键公式的含义。" in model.inputs[-1][-1]["content"]
    assert "甲甲甲" not in model.inputs[-1][-1]["content"]
    assert client.get(url).json()[1]["content"] == answer


@pytest.mark.parametrize("focus", ["同一步", ""])
def test_ambiguous_or_missing_feedback_focus_does_not_guess_excerpt(tmp_path, focus):
    answer = "同一步\n\n" + "甲" * 6000 + "\n\n同一步\n\n" + "乙" * 6000
    client, model, _, _, conv = setup(tmp_path, [{"role": "assistant", "content": answer}])
    client = TestClient(create_app(small_settings(tmp_path), model))
    url = f"/api/conversations/{conv['id']}/messages"
    first = client.post(url, json={"content": "解释"}).json()
    result = client.post(
        url,
        json={
            "content": "没懂",
            "feedback": {
                "kind": "stuck",
                "target_run_id": first["run_id"],
                "focus_text": focus,
            },
        },
    ).json()
    assert result["state"] == "failed" and len(model.inputs) == 1
    state = client.get(f"/api/conversations/{conv['id']}/learning").json()
    assert state["feedback"][-1]["target_run_id"] == first["run_id"]


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_preview_matches_encoded_cost_and_counts_native_opaque_items(tmp_path, protocol):
    Image.new("RGB", (30, 20), "blue").save(tmp_path / "test.png")
    provider = OpenAIProvider(Settings(_env_file=None, data_root=tmp_path, protocol=protocol))
    history = [
        {"role": "user", "content": "图中的颜色？", "attachments": [{"asset_path": "test.png"}]}
    ]
    if protocol == "responses":
        history.insert(
            0,
            {
                "role": "assistant",
                "content": "",
                "provider_items": [{"type": "reasoning", "encrypted_content": "opaque" * 100}],
            },
        )
    preview = provider.build_request(history, "system", [], encode_images=False)
    encoded = provider.build_request(history, "system", [])
    assert provider.estimate(preview) == provider.estimate(encoded)
    assert provider.estimate(preview)["images"] == 1
    assert provider.estimate(preview)["opaque_units"] > 0 if protocol == "responses" else True
    assert "asset_path" not in json.dumps(encoded.payload)


def native_tool_group():
    message = call("list_materials", {}, "native-call")
    message["provider_items"] = [
        {"type": "reasoning", "encrypted_content": "opaque" * 2000},
        {
            "type": "function_call",
            "name": "list_materials",
            "arguments": "{}",
            "call_id": "native-call",
        },
    ]
    return ProtocolGroup(
        "old:tools",
        "tools",
        [message, {"role": "tool", "tool_call_id": "native-call", "content": "{}"}],
    )


def test_native_batch_preserved_or_removed_as_unit_and_profile_change_uses_visible_only(tmp_path):
    settings = small_settings(tmp_path, protocol="responses")
    manager = ContextManager(settings)
    group = native_tool_group()
    original = copy.deepcopy(group.dump())
    turn = {
        "turn_seq": 1,
        "profile": manager.profile,
        "replay_status": "ready",
        "groups": [group.dump()],
        "visible": [{"role": "assistant", "content": "已交付的解释"}],
    }
    current = [ProtocolGroup("new:input", "input", [{"role": "user", "content": "追问"}])]
    result = manager.prepare([turn], current, TurnInput("追问"), "", [], True)
    assert result.messages[0]["provider_items"] == original["messages"][0]["provider_items"]
    small = Settings(
        _env_file=None,
        data_root=tmp_path,
        protocol="responses",
        context_budget_tokens=4000,
        context_output_reserve_tokens=256,
        context_headroom_tokens=256,
    )
    trimmed = ContextManager(small).prepare([turn], current, TurnInput("追问"), "", [], True)
    assert all("provider_items" not in m and m["role"] != "tool" for m in trimmed.messages)
    assert "历史工具索引" in json.dumps(trimmed.messages, ensure_ascii=False)
    assert group.dump() == original
    turn["profile"] = {**turn["profile"], "model": "another"}
    switched = manager.prepare([turn], current, TurnInput("追问"), "", [], True)
    assert "已交付的解释" in json.dumps(switched.messages, ensure_ascii=False)
    assert all("provider_items" not in m for m in switched.messages)


def test_incomplete_tool_batch_rejected_locally(tmp_path):
    group = ProtocolGroup("bad", "tools", [call("list_materials", {})])
    with pytest.raises(ContextError, match="不配对"):
        ContextManager(small_settings(tmp_path)).prepare([], [group], TurnInput("问"), "", [], True)


def test_legacy_migration_retains_full_original_and_uses_visible_messages_once(tmp_path):
    store = Store(tmp_path)
    conv = store.conversation(store.space("space")["id"])
    old = [
        {"role": "user", "content": "旧问题[旧注入不应复制]"},
        {
            "role": "assistant",
            "content": "old",
            "provider_items": [{"type": "reasoning", "encrypted_content": "secret-opaque"}],
        },
    ]
    run = store.start_run(conv["id"], "旧问题")
    store.finish_run(run, "completed", "旧答案", old)
    model = Scripted([{"role": "assistant", "content": "新答案"}])
    client = TestClient(create_app(small_settings(tmp_path), model))
    assert (
        client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "追问"}).json()[
            "state"
        ]
        == "completed"
    )
    seen = json.dumps(model.inputs[-1], ensure_ascii=False)
    assert (
        seen.count('"content": "旧问题"') == 1
        and "旧注入" not in seen
        and "secret-opaque" not in seen
    )
    store.context_turns(conv["id"])
    with store.connect() as db:
        legacy = db.execute("SELECT groups_json FROM protocol_turns WHERE turn_seq=0").fetchall()
    assert len(legacy) == 1 and json.loads(legacy[0][0])[0]["messages"] == old


def test_same_turn_image_restore_reuses_asset_and_duplicate_guard_still_stops(
    tmp_path, monkeypatch
):
    client, space, material, conv, model = pdf_setup(tmp_path)
    text = client.post(
        f"/api/spaces/{space}/materials", files={"file": ("long.txt", b"x" * 7000)}
    ).json()["id"]
    args = {"material_id": material, "page": 2, "region": None}
    model.responses = [
        call("inspect_page", args, "view"),
        call("read_passage", {"material_id": text, "start_line": 1, "end_line": 1}, "text"),
        call("inspect_page", args, "restore"),
        call("inspect_page", args, "duplicate"),
        {"role": "assistant", "content": "图像已经重新附加。"},
    ]
    executions = []
    execute = MaterialTools.execute

    def tracked(self, name, arguments, **kwargs):
        executions.append(name)
        return execute(self, name, arguments, **kwargs)

    monkeypatch.setattr(MaterialTools, "execute", tracked)
    settings = Settings(
        _env_file=None,
        data_root=tmp_path,
        context_budget_tokens=26000,
        context_output_reserve_tokens=512,
        context_headroom_tokens=512,
    )
    client = TestClient(create_app(settings, model))
    result = client.post(
        f"/api/conversations/{conv}/messages", json={"content": "对照图与笔记解释"}
    ).json()
    assert result["state"] == "completed"
    assert not any(m.get("attachments") for m in model.inputs[2])
    assert any(m.get("attachments") for m in model.inputs[3])
    observations = [
        e["observation"]
        for e in client.get(f"/api/runs/{result['run_id']}").json()["events"]
        if e["type"] == "tool"
    ]
    assert observations[2]["context_reattached"] and observations[3]["error"] == "duplicate_call"
    assert executions.count("inspect_page") == 1


def test_missing_old_image_becomes_notice_but_current_valid_image_is_retained(tmp_path):
    client, _space, material, conv, model = pdf_setup(tmp_path)
    model.responses = [
        call("inspect_page", {"material_id": material, "page": 2, "region": None}),
        {"role": "assistant", "content": "图的解释"},
        {"role": "assistant", "content": "旧图片当前不可用。"},
    ]
    url = f"/api/conversations/{conv}/messages"
    assert client.post(url, json={"content": "看图"}).json()["state"] == "completed"
    attachment = next(m["attachments"][0] for m in model.inputs[1] if m.get("attachments"))
    (tmp_path / attachment["asset_path"]).unlink()
    result = client.post(url, json={"content": "继续"}).json()
    assert result["state"] == "completed" and result["context_limited"]
    assert not any(m.get("attachments") for m in model.inputs[-1])
    assert "image_unavailable" in json.dumps(model.inputs[-1])


def test_corrupt_current_image_is_tool_observation_and_does_not_leave_running_lock(tmp_path):
    client, space, material, conv, model = pdf_setup(tmp_path)
    asset = Store(tmp_path).page_image(space, material, 2)
    (tmp_path / asset["asset_path"]).write_bytes(b"corrupt cached png")
    model.responses = [
        call("inspect_page", {"material_id": material, "page": 2, "region": None}),
        {"role": "assistant", "content": "当前图像不可用，无法核实视觉细节。"},
    ]
    result = client.post(f"/api/conversations/{conv}/messages", json={"content": "看图"}).json()
    assert result["state"] == "completed"
    assert json.loads(model.inputs[-1][-1]["content"])["error"] == "asset_unavailable"
    assert Store(tmp_path).run(result["run_id"])["state"] != "running"


def test_request_byte_limit_prevents_sending_required_large_image(tmp_path):
    import random

    Image.frombytes("RGB", (128, 128), random.Random(42).randbytes(128 * 128 * 3)).save(
        tmp_path / "noise.png"
    )
    settings = small_settings(tmp_path, context_max_request_bytes=4096)
    group = ProtocolGroup(
        "new:input",
        "input",
        [{"role": "user", "content": "看图", "attachments": [{"asset_path": "noise.png"}]}],
    )
    with pytest.raises(ContextError) as error:
        ContextManager(settings).prepare([], [group], TurnInput("看图"), "", [], True)
    assert error.value.snapshot["estimate"]["request_bytes"] > 4096


def test_context_settings_validate_protocol_output_field_and_picture_capacity(tmp_path):
    with pytest.raises(ValueError):
        small_settings(tmp_path, max_context_images=1)
    with pytest.raises(ValueError):
        small_settings(tmp_path, output_token_field="max_output_tokens")
    settings = small_settings(tmp_path, output_token_field="max_completion_tokens")
    request = OpenAIProvider(settings).build_request([], "", [])
    assert request.payload["max_completion_tokens"] == 512
