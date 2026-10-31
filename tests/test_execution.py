"""Real process restart, operation replay and bounded recovery tests for RT2."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_feedback import deliver
from test_learning import Scripted
from test_runtime import eventually

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import ProtocolGroup, profile_id, validate_group
from learning_agent.execution_store import execution_profile
from learning_agent.provider import ModelError
from learning_agent.retrieval import Embedder, Retriever
from learning_agent.store import Store


def prepared(root, scenario="batch", **changes):
    settings = Settings(_env_file=None, data_root=root, **changes)
    store = Store(root)
    space = store.space("执行恢复验收")
    conv = store.conversation(space["id"])
    if scenario == "image":
        from test_pdf import pdf_bytes

        store.add_pdf(space["id"], "course.pdf", pdf_bytes())
    else:
        store.add_material(
            space["id"], "notes.txt", "近似的误差需要足够小。\n不能把近似当作严格相等。"
        )
    if scenario.startswith("summary"):
        store.context_turns(conv["id"])
        for index in range(8):
            question = f"第{index}次近似条件的问题"
            answer = "明确误差与近似前提。" * 1800
            old = store.start_run(conv["id"], question)
            groups = [
                ProtocolGroup(
                    old + ":input", "input", [{"role": "user", "content": question}]
                ).dump(),
                ProtocolGroup(
                    old + ":answer", "answer", [{"role": "assistant", "content": answer}]
                ).dump(),
            ]
            store.finish_run(
                old,
                "completed",
                answer,
                [],
                archive={
                    "profile": profile_id(settings),
                    "input": {"question": question},
                    "groups": groups,
                },
            )
    (root / "settings.json").write_text(json.dumps({**changes, "data_root": str(root)}))
    if scenario.startswith("embedding"):

        class Vectors:
            key = Embedder(settings).key

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        Retriever(store, settings, Vectors()).prepare(space["id"], raise_errors=True)
    rid = store.submit_run(
        conv["id"],
        {"client_request_id": "request-0001", "content": "请解释近似"},
        profile_id(settings),
        20,
        execution=execution_profile(settings),
    )
    return store, settings, conv, rid


@contextmanager
def worker(root, scenario="batch", target=""):
    env = {**os.environ, "RT_ROOT": str(root), "RT_SCENARIO": scenario, "RT_TARGET": target}
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("runtime_fault_worker.py"))],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def barrier(root, process):
    def ready():
        if process.poll() is not None:
            raise AssertionError(process.stderr.read().decode())
        return (root / "barrier").exists()

    eventually(ready, 10)


def calls(root):
    path = root / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize(
    "target,first_calls", [("prepared", 0), ("returned", 1), ("tool_1", 1), ("ready", 2)]
)
def test_restart_reuses_prepared_returned_partial_batch_and_ready_answer(
    tmp_path, target, first_calls
):
    store, _, conv, rid = prepared(tmp_path)
    with worker(tmp_path, target=target) as process:
        barrier(tmp_path, process)
        assert len(calls(tmp_path)) == first_calls
        before = store.run(rid)["remaining_seconds"]
    with worker(tmp_path) as process:
        eventually(lambda: store.run(rid)["state"] == "completed", 10)
        assert process.poll() is None
        assert len(calls(tmp_path)) == 2
        assert calls(tmp_path)[-1]["tool_ids"] == ["read-1", "read-2"]
        assert store.run(rid)["recovery_count"] == 1
        assert store.run(rid)["remaining_seconds"] <= 180
        if target == "ready":
            assert store.run(rid)["remaining_seconds"] == before
        assert len(store.messages(conv["id"])) == 2
        turns = store.context_turns(conv["id"])
        assert len(turns) == 1 and turns[0]["replay_status"] == "ready"
        for group in turns[0]["groups"]:
            validate_group(group)
        assert store.operation_count(rid, "tool") == 3
        assert store.current_learning_state(conv["id"])["revision"] == 1


def test_unknown_main_waits_for_explicit_idempotent_retry(tmp_path):
    store, settings, conv, rid = prepared(tmp_path)
    with worker(tmp_path, target="sending") as process:
        barrier(tmp_path, process)
    with worker(tmp_path):
        eventually(lambda: store.run(rid)["state"] == "interrupted", 10)
        assert len(calls(tmp_path)) == 1
        assert store.public_run(rid)["allowed_actions"] == ["cancel", "retry_model"]
        assert store.run(rid)["remaining_seconds"] <= 90
    model = Scripted([deliver("明确重试后完成")])
    with TestClient(create_app(settings, model)) as client:
        revision = client.get(f"/api/runs/{rid}").json()["revision"]
        payload = {"action_id": "retry-action-0001", "expected_revision": revision}
        assert client.post(f"/api/runs/{rid}/resume", json=payload).status_code == 409
        payload["acknowledge_unknown_request"] = True
        accepted = client.post(f"/api/runs/{rid}/resume", json=payload)
        assert accepted.status_code == 202, accepted.text
        assert client.post(f"/api/runs/{rid}/resume", json=payload).json() == accepted.json()
        eventually(lambda: store.run(rid)["state"] == "completed")
        assert len(model.inputs) == 1 and store.operation_count(rid, "main_model") == 2
        assert store.run(rid)["recovery_count"] == 1
        assert len(store.messages(conv["id"])) == 2
        different = {**payload, "acknowledge_unknown_request": False}
        assert client.post(f"/api/runs/{rid}/resume", json=different).status_code == 409


@pytest.mark.parametrize(
    "target,status",
    [("embedding_sending", "remote_result_unknown:bm25_fallback"), ("embedding_returned", "ready")],
)
def test_query_vector_restart_reuses_result_or_degrades_without_resending(tmp_path, target, status):
    scenario = "embedding"
    store, _, _, rid = prepared(
        tmp_path,
        scenario,
        embedding_provider="compatible",
        embedding_base_url="http://unused.invalid/v1",
    )
    with worker(tmp_path, scenario, target) as process:
        barrier(tmp_path, process)
    with worker(tmp_path, scenario):
        eventually(lambda: store.run(rid)["state"] == "completed", 10)
        assert [c["kind"] for c in calls(tmp_path)].count("embedding") == 1
        observations = [
            e["observation"]
            for e in store.run(rid)["events"]
            if e["type"] == "tool" and e["name"] == "search_materials"
        ]
        assert observations[0]["dense_status"] == status
        assert store.operation_count(rid, "query_embedding") == 1
        assert (
            store.operation_count(rid, "tool") == 3
        )  # interrupted search attempt, replay, delivery


def test_observed_source_change_refuses_restart_instead_of_using_stale_result(tmp_path):
    store, _, _, rid = prepared(tmp_path)
    with worker(tmp_path, target="tool_1") as process:
        barrier(tmp_path, process)
    with store.connect() as db:
        db.execute("UPDATE materials SET revision=revision+1")
    with worker(tmp_path):
        eventually(lambda: store.run(rid)["state"] == "interrupted", 10)
        eventually(lambda: store.run(rid)["recovery_reason"] == "context_changed")
        assert store.public_run(rid)["allowed_actions"] == ["cancel"]
        assert len(calls(tmp_path)) == 1


def test_missing_image_is_regenerated_without_repeating_completed_tool_or_model(tmp_path):
    store, _, conv, rid = prepared(tmp_path, "image")
    with worker(tmp_path, "image", "second_model_prepared") as process:
        barrier(tmp_path, process)
    image = next(
        m["attachments"][0]
        for m in store.checkpoint(rid)["payload"]["prepared"]["messages"]
        if m.get("attachments")
    )
    (tmp_path / image["asset_path"]).unlink()
    with worker(tmp_path, "image"):
        eventually(lambda: store.run(rid)["state"] == "completed", 10)
        from learning_agent.provider import asset_info

        assert asset_info(tmp_path, image)["width"] > 0
        assert len(calls(tmp_path)) == 2
        assert store.operation_count(rid, "tool") == 3  # initial view, regeneration, delivery
        assert store.checkpoint(rid)["payload"]["images_used"] == 2
        assert any(e["type"] == "asset.restored" for e in store.progress_events(rid))
        assert len(store.messages(conv["id"])) == 2


@pytest.mark.parametrize(
    "target,expected", [("summary_sending", "failed"), ("summary_returned", "ready")]
)
def test_summary_unknown_degrades_and_returned_response_is_parsed_without_resend(
    tmp_path, target, expected
):
    settings = {
        "context_budget_tokens": 18000,
        "context_output_reserve_tokens": 512,
        "context_headroom_tokens": 512,
    }
    store, _, conv, rid = prepared(tmp_path, "summary", **settings)
    with worker(tmp_path, "summary", target) as process:
        barrier(tmp_path, process)
    with worker(tmp_path, "summary"):
        eventually(lambda: store.run(rid)["state"] == "completed", 10)
        assert [c["kind"] for c in calls(tmp_path)].count("summary") == 1
        assert store.operation_count(rid, "summary") == 1
        summary = store.latest_summary_attempt(
            conv["id"], 0, profile_id(Settings(_env_file=None, data_root=tmp_path, **settings))
        )
        assert summary["status"] == expected
        assert len(store.messages(conv["id"])) == 18


def test_recovery_attempt_limit_and_prepared_reservation_survive_repeated_deaths(tmp_path):
    store, _, _, rid = prepared(tmp_path)
    for recovery in range(4):
        (tmp_path / "barrier").unlink(missing_ok=True)
        with worker(tmp_path, target="prepared") as process:
            barrier(tmp_path, process)
            assert store.run(rid)["recovery_count"] == recovery
            assert store.run(rid)["remaining_seconds"] == 90
    with worker(tmp_path):
        eventually(lambda: store.run(rid)["recovery_reason"] == "recovery_limit", 10)
        assert store.public_run(rid)["allowed_actions"] == ["cancel"]
        assert store.operation_count(rid, "main_model") == 1
        assert calls(tmp_path) == []


def test_unknown_request_time_reservation_is_not_refunded_or_reset_by_retry(tmp_path):
    store, settings, _, rid = prepared(tmp_path, "network_unknown")
    with worker(tmp_path, "network_unknown"):
        eventually(lambda: store.run(rid)["state"] == "interrupted", 10)
    model = Scripted([ModelError("未知结果", outcome="unknown")])
    with TestClient(create_app(settings, model)) as client:
        public = client.get(f"/api/runs/{rid}").json()
        assert public["remaining_seconds"] == 90
        result = client.post(
            f"/api/runs/{rid}/resume",
            json={
                "action_id": "budget-retry-0001",
                "expected_revision": public["revision"],
                "acknowledge_unknown_request": True,
            },
        )
        assert result.status_code == 202, result.text
        eventually(lambda: store.run(rid)["state"] == "interrupted")
        public = client.get(f"/api/runs/{rid}").json()
        assert public["remaining_seconds"] == 0 and public["allowed_actions"] == ["cancel"]
        assert len(model.inputs) == 1 and store.operation_count(rid, "main_model") == 2
        with store.connect() as db:
            assert [
                tuple(r)
                for r in db.execute(
                    "SELECT state,usage_status FROM run_operations WHERE purpose='main_model'"
                )
            ] == [("unknown", "unknown"), ("unknown", "unknown")]


def slow_render_code(marker):
    return """
import os,sys,json,time,subprocess,runpy
from pathlib import Path
import learning_agent.pdf as pdf
def stalled(*args,**kwargs):
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    Path(MARKER).write_text(json.dumps({'renderer':os.getpid(),'descendant':child.pid}))
    (Path(sys.argv[1])/'incomplete.png').write_bytes(b'partial png')
    time.sleep(60)
pdf.render_page=stalled
runpy.run_module('learning_agent.render_process',run_name='__main__')
""".replace("MARKER", repr(str(marker)))


def no_process(pid):
    # A killed descendant can briefly be a zombie awaiting init; it cannot continue writing.
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    assert not status.stderr, status.stderr
    return not status.stdout.strip() or status.stdout.strip().startswith("Z")


def test_cancel_render_kills_owned_process_group_and_never_publishes_partial_png(
    tmp_path, monkeypatch
):
    import asyncio

    from test_learning import call
    from test_pdf import prepared as pdf_client

    from learning_agent import render_process

    client, _space, material, conv, model = pdf_client(tmp_path)
    model.responses = [call("inspect_page", {"material_id": material, "page": 1, "region": None})]
    original = asyncio.create_subprocess_exec
    marker = tmp_path / "render-processes.json"

    async def spawn(*args, **kwargs):
        if "learning_agent.render_process" in args:
            args = (sys.executable, "-c", slow_render_code(marker), args[-1])
        return await original(*args, **kwargs)

    monkeypatch.setattr(render_process.asyncio, "create_subprocess_exec", spawn)
    result = client.post(
        f"/api/conversations/{conv}/runs",
        json={"content": "查看页面", "client_request_id": "render-cancel-0001"},
    )
    rid = result.json()["run_id"]
    eventually(marker.exists)
    processes = json.loads(marker.read_text())
    client.post(f"/api/runs/{rid}/cancel")
    store = client.app.state.store
    eventually(lambda: store.run(rid)["state"] == "cancelled")
    eventually(lambda: all(no_process(pid) for pid in processes.values()))
    assert not list((tmp_path / "materials" / material).glob("page-*.png"))
    assert not list((tmp_path / ".runtime-attempts").glob("**/*.png"))
    assert store.current_learning_state(conv)["revision"] == 0


def test_parent_death_pipe_terminates_render_descendants_before_publication(tmp_path):
    store, _, _, _ = prepared(tmp_path, "image")
    space = store.spaces()[0]["id"]
    material = store.materials(space)[0]["id"]
    attempt = tmp_path / ".runtime-attempts" / "parent-death"
    attempt.mkdir(parents=True)
    marker = tmp_path / "parent-render.json"
    request = {
        "root": str(tmp_path),
        "space_id": space,
        "vision": True,
        "name": "inspect_page",
        "arguments": {"material_id": material, "page": 1, "region": None},
        "image_budget": 3,
    }
    parent_code = (
        """
import subprocess,sys,time,json
process=subprocess.Popen([sys.executable,'-c',CHILD,ATTEMPT],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
process.stdin.write((json.dumps(REQUEST)+'\\n').encode());process.stdin.flush()
time.sleep(60)
""".replace("CHILD", repr(slow_render_code(marker)))
        .replace("ATTEMPT", repr(str(attempt)))
        .replace("REQUEST", repr(request))
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    try:
        eventually(marker.exists)
        processes = json.loads(marker.read_text())
    finally:
        parent.kill()
        parent.wait(timeout=5)
    eventually(lambda: all(no_process(pid) for pid in processes.values()))
    assert not (tmp_path / "materials" / material / "page-1-full.png").exists()
    assert (attempt / "incomplete.png").exists()  # orphan staging is never adopted as a cache


def test_image_reservation_survives_death_before_tool_result(tmp_path):
    store, _, _, rid = prepared(tmp_path, "image", max_images_per_turn=1)
    with worker(tmp_path, "image", "tool_sending") as process:
        barrier(tmp_path, process)
    assert store.image_usage(rid) == 1
    with worker(tmp_path, "image"):
        eventually(lambda: store.run(rid)["state"] == "completed", 10)
        assert store.image_usage(rid) == 1
        assert store.checkpoint(rid)["payload"]["images_used"] == 1
        assert not list((tmp_path / "materials").glob("**/page-*.png"))
        assert any(
            e.get("observation", {}).get("error") == "image_budget_exhausted"
            for e in store.run(rid)["events"]
        )


def test_responses_native_items_survive_returned_boundary_and_never_enter_progress(tmp_path):
    store, _, _, rid = prepared(tmp_path, protocol="responses")
    with worker(tmp_path, target="returned") as process:
        barrier(tmp_path, process)
    with worker(tmp_path):
        eventually(lambda: store.run(rid)["state"] == "completed", 10)
        assert len(calls(tmp_path)) == 2
        checkpoint = store.checkpoint(rid)["payload"]
        assert (
            checkpoint["groups"][1]["messages"][0]["provider_items"][0]["encrypted_content"]
            == "private-opaque"
        )
        assert "private-opaque" not in json.dumps(store.progress_events(rid))
        assert "private-opaque" not in json.dumps(store.public_run(rid))


@pytest.mark.parametrize(
    "http_status,body,expected,expected_requests",
    [
        (
            200,
            {"choices": [{"message": {"content": 42}}], "usage": {"total_tokens": 7}},
            "returned_invalid",
            1,
        ),
        (503, {"error": "private upstream body"}, "failed", 2),
    ],
)
def test_known_provider_failure_is_not_misclassified_as_unknown_retry(
    tmp_path, http_status, body, expected, expected_requests
):
    import httpx
    from pydantic import SecretStr

    from learning_agent.provider import OpenAIProvider

    settings = Settings(_env_file=None, data_root=tmp_path, api_key=SecretStr("offline-only"))
    sent = []

    def response(request):
        sent.append(request)
        return httpx.Response(http_status, json=body)

    with TestClient(
        create_app(settings, OpenAIProvider(settings, httpx.MockTransport(response)))
    ) as client:
        sid = client.post("/api/spaces", json={"content": "测试"}).json()["id"]
        cid = client.post(f"/api/spaces/{sid}/conversations").json()["id"]
        rid = client.post(
            f"/api/conversations/{cid}/runs",
            json={"content": "解释", "client_request_id": "failure-0001"},
        ).json()["run_id"]
        store = client.app.state.store
        eventually(lambda: store.run(rid)["state"] == "failed")
        assert len(sent) == expected_requests and store.public_run(rid)["allowed_actions"] == []
        with store.connect() as db:
            row = db.execute(
                "SELECT state,result,usage_status FROM run_operations WHERE run_id=?", (rid,)
            ).fetchone()
        assert row["state"] == expected
        if expected == "returned_invalid":
            assert (
                json.loads(row["result"])["usage"]["total_tokens"] == 7
                and row["usage_status"] == "known"
            )
        assert "private upstream body" not in json.dumps(store.public_run(rid))


def test_final_transaction_rechecks_source_versions_before_answer_and_learning_commit(
    tmp_path, monkeypatch
):
    client, model, _, material, conv = __import__("test_learning").setup(tmp_path, [])
    from test_learning import call

    model.responses = [
        call("read_passage", {"material_id": material["id"], "start_line": 1, "end_line": 1}),
        deliver("不应交付过期归因"),
    ]
    store = client.app.state.store
    original = store.finish_run

    def changed(*args, **kwargs):
        with store.connect() as db:
            db.execute("UPDATE materials SET revision=revision+1 WHERE id=?", (material["id"],))
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "finish_run", changed)
    rid = client.post(
        f"/api/conversations/{conv['id']}/runs",
        json={"content": "解释资料", "client_request_id": "source-race-0001"},
    ).json()["run_id"]
    eventually(lambda: store.run(rid)["state"] == "interrupted")
    assert len(store.messages(conv["id"])) == 1
    assert store.current_learning_state(conv["id"])["revision"] == 0
    assert store.context_turns(conv["id"]) == []
