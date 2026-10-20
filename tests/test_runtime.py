"""RT1 lifecycle tests: durable identity, cancellation, isolation and real process death."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient as BaseClient
from test_feedback import deliver
from test_learning import call, setup
from test_memory import attach, save, topic

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import profile_id
from learning_agent.run_store import RuntimeConflict
from learning_agent.store import Store
from learning_agent.tools import MaterialTools


def eventually(function, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = function()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("运行状态未在期限内到达")


def submit(client, conv, identity="request-0001", content="解释近似", **extra):
    return client.post(
        f"/api/conversations/{conv['id']}/runs",
        json={"client_request_id": identity, "content": content, **extra},
    )


def ended(client, run_id, state="completed"):
    return eventually(
        lambda: r if (r := client.get(f"/api/runs/{run_id}").json())["state"] == state else None
    )


class GateModel:
    def __init__(self, result=None):
        self.started, self.release, self.cancelled = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.calls = 0
        self.result = result or deliver("近似允许有受控的小误差。")

    async def step(self, *_args, **_kwargs):
        self.calls += 1
        self.started.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.01)
            return self.result, {"total_tokens": 10}
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def gated(tmp_path, result=None):
    client, _, space, material, conv = setup(tmp_path, [])
    model = GateModel(result)
    client.app.state.runtime.provider = model
    return client, model, space, material, conv


def test_durable_submit_idempotency_refresh_and_pending_edits(tmp_path):
    client, model, _, _, conv = gated(tmp_path)
    first = submit(client, conv)
    assert first.status_code == 202
    rid = first.json()["run_id"]
    assert model.started.wait(2)
    assert submit(client, conv).json()["run_id"] == rid
    assert submit(client, conv, content="另一个问题").status_code == 409
    assert submit(client, conv, identity="request-0002").status_code == 409
    # The refresh query finds the same task without touching its execution.
    active = client.get(f"/api/conversations/{conv['id']}/runs?status=active").json()
    assert [r["id"] for r in active] == [rid]
    assert "owner_token" not in active[0] and active[0]["recoverable"] is False
    assert (
        client.patch(
            f"/api/conversations/{conv['id']}/learning", json={"revision": 0, "goal": "改目标"}
        ).status_code
        == 409
    )
    model.release.set()
    assert ended(client, rid)["answer"] == "近似允许有受控的小误差。"
    assert submit(client, conv).json()["run_id"] == rid
    assert model.calls == 1
    messages = client.get(f"/api/conversations/{conv['id']}/messages").json()
    assert len(messages) == 2
    assert (
        client.get(f"/api/conversations/{conv['id']}/runs?before={active[0]['cursor']}").json()
        == []
    )


def test_queue_limit_queued_cancel_and_running_cancel_are_durable(tmp_path):
    client, model, space, _, conv = gated(tmp_path)
    client.app.state.runtime.settings.runtime_queue_limit = 1
    rid = submit(client, conv).json()["run_id"]
    assert model.started.wait(2)
    second = client.post(f"/api/spaces/{space['id']}/conversations").json()
    third = client.post(f"/api/spaces/{space['id']}/conversations").json()
    queued = submit(client, second).json()["run_id"]
    assert client.get(f"/api/runs/{queued}").json()["state"] == "queued"
    assert submit(client, third).status_code == 503
    assert client.get(f"/api/conversations/{third['id']}/messages").json() == []
    assert client.post(f"/api/runs/{queued}/cancel").json()["state"] == "cancelled"
    assert client.post(f"/api/runs/{rid}/cancel").json()["state"] in {"cancelling", "cancelled"}
    ended(client, rid, "cancelled")
    assert model.cancelled.is_set() and model.calls == 1
    assert client.post(f"/api/runs/{rid}/cancel").json()["state"] == "cancelled"
    store = client.app.state.store
    assert store.current_learning_state(conv["id"])["revision"] == 0
    assert all(t["replay_status"] == "failed" for t in store.context_turns(conv["id"]))
    assert len(client.get(f"/api/conversations/{conv['id']}/messages").json()) == 2
    model.release.set()
    next_run = submit(client, conv, identity="request-0002").json()["run_id"]
    ended(client, next_run)
    assert model.calls == 2


def test_cancel_drains_started_thread_and_never_runs_next_model(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = MaterialTools.execute

    def blocked(self, *args, **kwargs):
        started.set()
        assert release.wait(3)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MaterialTools, "execute", blocked)
    client, model, _, material, conv = setup(tmp_path, [])
    model.responses = [
        call("read_passage", {"material_id": material["id"], "start_line": 1, "end_line": 2})
    ]
    rid = submit(client, conv).json()["run_id"]
    assert started.wait(2)
    client.post(f"/api/runs/{rid}/cancel")
    assert client.get(f"/api/runs/{rid}").json()["state"] == "cancelling"
    release.set()
    ended(client, rid, "cancelled")
    assert len(model.inputs) == 1
    events = client.get(f"/api/runs/{rid}/events").json()
    assert any(e["type"] == "tool.started" for e in events)
    assert not any(e["type"] == "tool.finished" for e in events)


def test_events_cursor_sse_replay_and_supervisor_failure_isolation(tmp_path):
    client, model, space, _, conv = setup(
        tmp_path, [RuntimeError("private-secret"), deliver("恢复正常")]
    )
    rid = submit(client, conv).json()["run_id"]
    failed = ended(client, rid, "failed")
    assert "private-secret" not in json.dumps(failed)
    second = client.post(f"/api/spaces/{space['id']}/conversations").json()
    next_rid = submit(client, second).json()["run_id"]
    ended(client, next_rid)
    events = client.get(f"/api/runs/{rid}/events").json()
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "run.failed"
    replay = client.get(
        f"/api/runs/{rid}/events?after=999",
        headers={"Last-Event-ID": "1", "Accept": "text/event-stream"},
    )
    assert replay.status_code == 200 and "id: 1\n" not in replay.text
    assert "event: progress" in replay.text and "run.failed" in replay.text
    assert "owner_token" not in replay.text and "private-secret" not in replay.text
    assert client.get(f"/api/runs/{rid}/events?after=999").status_code == 422
    assert (
        client.get(f"/api/runs/{rid}/events", headers={"Last-Event-ID": "bad"}).status_code == 422
    )
    assert len(model.inputs) == 2


def test_store_construction_does_not_recover_and_second_scheduler_is_rejected(tmp_path):
    client, model, _, _, conv = gated(tmp_path)
    rid = submit(client, conv).json()["run_id"]
    assert model.started.wait(2)
    assert Store(tmp_path).public_run(rid)["state"] == "running"
    with (
        pytest.raises(RuntimeError, match="已有问答调度器"),
        BaseClient(create_app(Settings(_env_file=None, data_root=tmp_path), model)),
    ):
        pass
    assert client.get(f"/api/runs/{rid}").json()["state"] == "running"
    client.post(f"/api/runs/{rid}/cancel")
    ended(client, rid, "cancelled")


def test_no_lifespan_returns_unavailable_and_request_disconnect_keeps_runtime(tmp_path):
    model = GateModel()
    app = create_app(Settings(_env_file=None, data_root=tmp_path), model)
    space = app.state.store.space("学习")
    conv = app.state.store.conversation(space["id"])

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            result = await client.post(
                f"/api/conversations/{conv['id']}/runs",
                json={"content": "问题", "client_request_id": "request-0001"},
            )
            assert result.status_code == 503
            async with app.router.lifespan_context(app):
                pending = asyncio.create_task(
                    client.post(
                        f"/api/conversations/{conv['id']}/messages",
                        json={"content": "问题"},
                        headers={"Idempotency-Key": "legacy-0001"},
                    )
                )
                for _ in range(200):
                    if model.started.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert model.started.is_set()
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                assert not model.cancelled.is_set()
                model.release.set()
                for _ in range(200):
                    runs = (await client.get(f"/api/conversations/{conv['id']}/runs")).json()
                    if runs[0]["state"] == "completed":
                        break
                    await asyncio.sleep(0.01)
                assert runs[0]["state"] == "completed" and model.calls == 1

    asyncio.run(scenario())


def frozen_store(tmp_path):
    store = Store(tmp_path)
    conv = store.conversation(store.space("学习")["id"])
    settings = Settings(_env_file=None, data_root=tmp_path)
    rid = store.submit_run(
        conv["id"],
        {"content": "问题", "client_request_id": "request-0001"},
        profile_id(settings),
        20,
    )
    return store, conv, rid, settings


def test_completion_cancel_transaction_winner_cas_and_exactly_once(tmp_path):
    store, conv, rid, _ = frozen_store(tmp_path)
    _, owner = store.claim_run()
    archive = {"profile": {}, "input": {"question": "问题"}, "groups": []}
    with pytest.raises(RuntimeConflict):
        store.finish_run(
            rid, "completed", "答案", [], archive=archive, expected_revision=99, owner_token=owner
        )
    assert store.public_run(rid)["state"] == "running"
    assert len(store.messages(conv["id"])) == 1
    store.cancel_run(rid)
    result = store.finish_run(rid, "completed", "不应交付", [], archive=archive, owner_token=owner)
    assert result["state"] == "cancelled"
    assert "不应交付" not in json.dumps(store.messages(conv["id"]), ensure_ascii=False)
    assert store.finish_run(rid, "completed", "重复答案", [], archive=archive) == result
    assert len(store.messages(conv["id"])) == 2
    assert len(store.context_turns(conv["id"])) == 1
    rid2 = store.submit_run(
        conv["id"], {"content": "问题2", "client_request_id": "request-0002"}, {}, 20
    )
    _, owner2 = store.claim_run()
    store.finish_run(rid2, "completed", "先完成", [], archive=archive, owner_token=owner2)
    store.cancel_run(rid2)
    assert store.public_run(rid2)["state"] == "completed"
    assert store.progress_events(rid2)[-1]["type"] == "run.completed"


def test_forget_scrubs_frozen_runtime_memory(tmp_path):
    client, _, space, _, conv = setup(tmp_path, [deliver("举例")])
    selected = topic(client, space)
    memory = save(client, space, selected, "私人学习偏好")
    attach(client, conv, selected)
    rid = submit(client, conv).json()["run_id"]
    ended(client, rid)
    store = client.app.state.store
    assert memory["text"] in json.dumps(store.run_input(rid), ensure_ascii=False)
    response = client.delete(
        f"/api/spaces/{space['id']}/memories/{memory['id']}?revision={memory['revision']}"
    )
    assert response.status_code == 200
    assert store.run_input(rid)["memories"]["items"] == []


def test_cancelled_summary_cannot_publish_ready_and_pending_context_cannot_change(tmp_path):
    client, model, space, _, conv = gated(tmp_path)
    selected = topic(client, space)
    attach(client, conv, selected)
    rid = submit(client, conv).json()["run_id"]
    assert model.started.wait(2)
    store = client.app.state.store
    owner = store.run(rid)["owner_token"]
    client.post(f"/api/runs/{rid}/cancel")
    record = {
        "id": "cancelled-summary",
        "conversation_id": conv["id"],
        "covered_end": 1,
        "context_epoch": 0,
        "state_revision": 0,
        "profile": {},
        "status": "ready",
        "payload": {},
        "result": {},
        "created_at": "test",
    }
    store.save_summary(record, run_id=rid, owner_token=owner)
    assert record["status"] == "interrupted"
    ended(client, rid, "cancelled")
    assert store.latest_summary_attempt(conv["id"], 0, {}, ready=True) is None


def test_frozen_profile_mismatch_interrupts_queued_run_without_calling_model(tmp_path):
    client, model, space, _, conv = gated(tmp_path)
    first = submit(client, conv).json()["run_id"]
    assert model.started.wait(2)
    second = client.post(f"/api/spaces/{space['id']}/conversations").json()
    queued = submit(client, second).json()["run_id"]
    client.app.state.runtime.settings.model = "changed-model"
    model.release.set()
    ended(client, first)
    interrupted = ended(client, queued, "interrupted")
    assert interrupted["recovery_reason"] == "context_changed" and model.calls == 1
    assert submit(client, second, identity="request-0002").status_code == 409
    assert client.post(f"/api/runs/{queued}/cancel").json()["state"] == "cancelled"


def test_completed_identity_can_be_confirmed_even_when_provider_becomes_unconfigured(tmp_path):
    client, _, _, _, conv = setup(tmp_path, [deliver("解释")])
    rid = submit(client, conv).json()["run_id"]
    ended(client, rid)
    client.close()
    with BaseClient(create_app(Settings(_env_file=None, data_root=tmp_path))) as reopened:
        assert submit(reopened, conv).json()["run_id"] == rid
        assert submit(reopened, conv, identity="request-0002").status_code == 503


def test_sigkill_restart_marks_unknown_does_not_resend_and_continues_queue(tmp_path):
    store, conv, rid, settings = frozen_store(tmp_path)
    second = store.conversation(store.get_conversation(conv["id"])["space_id"])
    second_rid = store.submit_run(
        second["id"],
        {"content": "排队问题", "client_request_id": "request-0002"},
        profile_id(settings),
        20,
    )
    marker = tmp_path / "remote-calls.txt"
    code = """
import asyncio, os
from pathlib import Path
from learning_agent.app import create_app
from learning_agent.config import Settings
class Model:
    async def step(self, history, *_args, **kwargs):
        question = history[-1]["content"]
        with Path(os.environ["RT_MARKER"]).open("a") as file:
            file.write(question.splitlines()[0] + "\\n")
        if "排队问题" not in question:
            await asyncio.Event().wait()
        return {"role":"assistant", "content":"排队任务完成"}, {}
async def main():
    app = create_app(Settings(_env_file=None, data_root=Path(os.environ["RT_ROOT"])), Model())
    async with app.router.lifespan_context(app):
        await asyncio.Event().wait()
asyncio.run(main())
"""
    env = {**os.environ, "RT_ROOT": str(tmp_path), "RT_MARKER": str(marker)}
    process = subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        eventually(lambda: marker.exists() and marker.read_text(), timeout=10)
        assert store.public_run(rid)["state"] == "running"
    finally:
        process.kill()
        process.communicate(timeout=5)
    process = subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        eventually(lambda: store.public_run(second_rid)["state"] == "completed", timeout=10)
        assert store.public_run(rid)["state"] == "interrupted"
        assert store.public_run(rid)["allowed_actions"] == ["cancel"]
        assert len(marker.read_text().splitlines()) == 2
        with store.connect() as db:
            assert (
                db.execute("SELECT state FROM run_operations WHERE run_id=?", (rid,)).fetchone()[0]
                == "unknown"
            )
        store.cancel_run(rid)
        assert store.public_run(rid)["state"] == "cancelled"
    finally:
        process.kill()
        process.communicate(timeout=5)
