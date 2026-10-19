import asyncio
import json

import httpx
from fastapi.testclient import TestClient
from test_feedback import deliver
from test_learning import call, setup

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import profile_id
from learning_agent.store import Store


def topic(client, space, materials=(), title="归一化"):
    response = client.post(
        f"/api/spaces/{space['id']}/learning-topics",
        json={
            "title": title,
            "material_ids": list(materials),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def attach(client, conv, selected, enabled=True):
    url = f"/api/conversations/{conv['id']}/context-settings"
    state = client.get(url).json()
    result = client.patch(
        url,
        json={"revision": state["revision"], "topic_id": selected["id"], "memory_enabled": enabled},
    )
    assert result.status_code == 200, result.text
    return result.json()


def save(client, space, selected, text="先给数值例子，再解释符号", **changes):
    result = client.post(
        f"/api/spaces/{space['id']}/memories",
        json={
            "topic_id": selected["id"],
            "kind": "preference",
            "text": text,
            **changes,
        },
    )
    assert result.status_code == 201, result.text
    return result.json()


def ask(client, conv, question="请解释归一化", **changes):
    result = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": question, **changes}
    )
    assert result.status_code == 200, result.text
    assert result.json()["state"] == "completed", result.text
    return result.json()


def test_opt_in_same_name_topics_and_spaces_are_isolated_and_persist(tmp_path):
    client, model, space, material, conv = setup(
        tmp_path, [{"role": "assistant", "content": "解释"}] * 4
    )
    first, same_name = topic(client, space), topic(client, space)
    memory = save(client, space, first)
    ask(client, conv)
    assert memory["text"] not in json.dumps(model.inputs[-1], ensure_ascii=False)
    attach(client, conv, first)
    ask(client, conv)
    assert memory["text"] in json.dumps(model.inputs[-1], ensure_ascii=False)
    before = client.get(f"/api/conversations/{conv['id']}/context-settings").json()
    assert attach(client, conv, first) == before
    attach(client, conv, same_name)
    ask(client, conv)
    assert memory["text"] not in json.dumps(model.inputs[-1], ensure_ascii=False)
    other = client.post("/api/spaces", json={"content": "其他空间"}).json()
    other_conv = client.post(f"/api/spaces/{other['id']}/conversations").json()
    assert (
        client.patch(
            f"/api/conversations/{other_conv['id']}/context-settings",
            json={"revision": 0, "topic_id": first["id"], "memory_enabled": True},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/spaces/{other['id']}/learning-topics",
            json={"title": "课程", "material_ids": [material["id"]]},
        ).status_code
        == 404
    )
    reopened = TestClient(create_app(Settings(_env_file=None, data_root=tmp_path), model))
    assert (
        reopened.get(f"/api/conversations/{conv['id']}/context-settings").json()["topic_id"]
        == same_name["id"]
    )
    fresh = reopened.post(f"/api/spaces/{space['id']}/conversations").json()
    assert not reopened.get(f"/api/conversations/{fresh['id']}/context-settings").json()[
        "memory_enabled"
    ]


def test_scope_relevance_zero_matches_and_bounded_selection(tmp_path):
    client, _, space, material, conv = setup(tmp_path, [])
    selected = topic(client, space, [material["id"]])
    attach(client, conv, selected)
    scoped = save(client, space, selected, "近似的数值例子", material_ids=[material["id"]])
    unresolved = save(client, space, selected, "正则化惩罚项的作用", kind="unresolved")
    store = Store(tmp_path)
    assert store.recall_memories(conv["id"], "什么是梯度")["items"] == []
    matched = store.recall_memories(conv["id"], "请解释正则化惩罚项")["items"]
    assert [m["id"] for m in matched] == [unresolved["id"]]
    focused = store.recall_memories(conv["id"], "请解释近似", {"material_id": material["id"]})
    assert [m["id"] for m in focused["items"]] == [scoped["id"]]
    outside = client.post(
        f"/api/spaces/{space['id']}/materials", files={"file": ("other.md", b"other")}
    ).json()
    assert store.recall_memories(conv["id"], "近似", {"material_id": outside["id"]})["items"] == []
    for n in range(9):
        save(client, space, selected, "偏好" + str(n) + "举例" * 80)
    bundle = store.recall_memories(conv["id"], "正则化")
    assert len(bundle["items"]) <= 6
    assert sum(len(m["text"]) for m in bundle["items"]) <= 1800


def test_forget_invalidates_history_summaries_notes_and_versions_but_keeps_raw_chat(tmp_path):
    client, model, space, _, conv = setup(
        tmp_path,
        [
            deliver("沿用旧记忆：苹果火箭规则", topic="苹果火箭规则", confusion="苹果火箭规则"),
            deliver("请重新说明偏好", topic="新问题", confusion=""),
        ],
    )
    selected = topic(client, space)
    attach(client, conv, selected)
    memory = save(client, space, selected, "苹果火箭规则")
    learning_url = f"/api/conversations/{conv['id']}/learning"
    revision = client.get(learning_url).json()["revision"]
    assert (
        client.patch(
            learning_url, json={"revision": revision, "goal": "保留我明确填写的目标"}
        ).status_code
        == 200
    )
    first = ask(client, conv)
    store = Store(tmp_path)
    state = store.current_learning_state(conv["id"])
    store.save_summary(
        {
            "id": "summary",
            "conversation_id": conv["id"],
            "covered_end": 1,
            "context_epoch": state["context_epoch"],
            "state_revision": state["revision"],
            "profile": profile_id(Settings(_env_file=None)),
            "status": "ready",
            "payload": {
                "items": [{"text": "苹果火箭规则", "source_run_ids": [first["run_id"]]}],
                "manifest": [],
            },
            "result": {},
            "created_at": "2026-10-19T23:30:00+08:00",
        }
    )
    base = f"/api/spaces/{space['id']}/memories/{memory['id']}"
    changed = client.patch(base, json={"revision": 1, "text": "先说明假设"}).json()
    assert changed["revision"] == 2
    assert client.patch(base, json={"revision": 1, "text": "过期修改"}).status_code == 409
    assert client.delete(base + "?revision=2").status_code == 200
    assert client.patch(base, json={"revision": 3, "status": "active"}).status_code == 404
    latest = ask(client, conv, "现在继续讲解")
    assert "苹果火箭规则" not in json.dumps(model.inputs[-1], ensure_ascii=False)
    assert "先说明假设" not in json.dumps(model.inputs[-1], ensure_ascii=False)
    learning = client.get(f"/api/conversations/{conv['id']}/learning").json()
    assert learning["summary"] is None
    assert learning["state"]["goal"]["text"] == "保留我明确填写的目标"
    assert learning["state"]["goal_pinned"]
    assert all(a["run_id"] != first["run_id"] for a in learning["attempts"])
    assert "苹果火箭规则" in json.dumps(
        client.get(f"/api/conversations/{conv['id']}/messages").json(), ensure_ascii=False
    )
    with store.connect() as db:
        versions = db.execute(
            "SELECT payload FROM memory_versions WHERE memory_id=?", (memory["id"],)
        ).fetchall()
    assert len(versions) == 1 and "text" not in json.loads(versions[0][0])
    first_archive = next(
        t for t in store.context_turns(conv["id"]) if t["run_id"] == first["run_id"]
    )
    assert not first_archive["raw_input"]["derived_context"]["memories"]["items"]
    snap = client.get(f"/api/runs/{latest['run_id']}/context").json()[0]
    assert snap["memory"]["selected"] == [] and snap["memory"]["history_after"] == 1


def test_recalled_source_only_allows_linked_cross_conversation_read_and_not_verified_citation(
    tmp_path,
):
    client, model, space, _, conv = setup(tmp_path, [deliver("先检查分母再化简")])
    selected = topic(client, space)
    attach(client, conv, selected)
    first = ask(client, conv, "为什么分母不能是零")
    memory = save(client, space, selected, "分母为零仍未解释清楚", source_run_id=first["run_id"])
    new = client.post(f"/api/spaces/{space['id']}/conversations").json()
    attach(client, new, selected)
    model.responses = [
        call(
            "recall_learning_context",
            {
                "run_id": first["run_id"],
                "turn_seq": None,
                "role": "user",
                "start_char": 0,
                "max_chars": 1000,
            },
        ),
        deliver("先说明零没有乘法逆元 [notes.md:L1-L2]"),
    ]
    result = ask(client, new)
    run = client.get(f"/api/runs/{result['run_id']}").json()
    observation = next(
        e["observation"] for e in run["events"] if e.get("name") == "recall_learning_context"
    )
    assert observation["content"] == "为什么分母不能是零"
    assert not any(r.get("position_verified") for r in run["references"])
    client.patch(
        f"/api/spaces/{space['id']}/memories/{memory['id']}",
        json={"revision": 1, "status": "disabled"},
    )
    model.responses = [
        call(
            "recall_learning_context",
            {
                "run_id": first["run_id"],
                "turn_seq": None,
                "role": "user",
                "start_char": 0,
                "max_chars": 1000,
            },
        ),
        deliver("请提供原话"),
    ]
    denied = ask(client, new)
    events = client.get(f"/api/runs/{denied['run_id']}").json()["events"]
    assert (
        next(e["observation"] for e in events if e.get("name") == "recall_learning_context")[
            "error"
        ]
        == "historical_context_not_found"
    )


def test_material_replacement_excludes_stale_memory_and_its_old_echo(tmp_path):
    client, model, space, material, conv = setup(
        tmp_path, [deliver("旧字幕中的苹果火箭规则"), deliver("当前没有此记录")]
    )
    selected = topic(client, space, [material["id"]])
    attach(client, conv, selected)
    memory = save(client, space, selected, "苹果火箭规则", material_ids=[material["id"]])
    store = Store(tmp_path)
    assert store.recall_memories(conv["id"], "继续", {"material_id": material["id"]})["items"]
    ask(client, conv)
    with store.connect() as db:
        store.invalidate_index(db, material["id"])
    assert store.recall_memories(conv["id"], "继续", {"material_id": material["id"]})["items"] == []
    ask(client, conv)
    assert "苹果火箭规则" not in json.dumps(model.inputs[-1], ensure_ascii=False)
    revised = client.patch(
        f"/api/spaces/{space['id']}/memories/{memory['id']}",
        json={"revision": 1, "text": "新字幕已核对的讲解偏好"},
    ).json()
    assert revised["material_revisions"][material["id"]] == 1
    assert store.recall_memories(conv["id"], "继续", {"material_id": material["id"]})["items"]


def test_running_turn_blocks_memory_mutations_across_linked_conversations(tmp_path):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        class Waiting:
            async def step(self, *args):
                entered.set()
                await release.wait()
                return {"role": "assistant", "content": "解释"}, {}

        app = create_app(Settings(_env_file=None, data_root=tmp_path), Waiting())
        store = app.state.store
        space = store.space("学习")
        conv = store.conversation(space["id"])
        selected = store.create_learning_topic(space["id"], {"title": "主题", "material_ids": []})
        store.edit_context_settings(
            conv["id"], {"revision": 0, "topic_id": selected["id"], "memory_enabled": True}
        )
        memory = store.create_memory(
            space["id"],
            {
                "topic_id": selected["id"],
                "kind": "preference",
                "text": "先举例",
                "material_ids": [],
                "source_run_id": None,
            },
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(
                client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "继续"})
            )
            await entered.wait()
            blocked = await client.delete(
                f"/api/spaces/{space['id']}/memories/{memory['id']}?revision=1"
            )
            assert blocked.status_code == 409
            release.set()
            assert (await task).json()["state"] == "completed"
            assert (
                await client.delete(f"/api/spaces/{space['id']}/memories/{memory['id']}?revision=1")
            ).status_code == 200

    asyncio.run(exercise())
