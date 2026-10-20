"""RT3: protocol boundaries, durable drafts and cancellation, without paid calls."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest
from client_helpers import TestClient
from test_execution import prepared
from test_feedback import deliver
from test_runtime import ended, eventually, submit

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.provider import ModelError, OpenAIProvider
from learning_agent.run_store import RunStopped
from learning_agent.streaming import StreamAssembly, answer_prefix


def event(data):
    return (
        "data: "
        + (data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
        + "\n\n"
    ).encode()


def chat_events(message):
    if message.get("tool_calls"):
        call = message["tool_calls"][0]
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call["id"],
                                "function": {"name": call["function"]["name"], "arguments": ""},
                            }
                        ]
                    },
                }
            ]
        }
        args = call["function"]["arguments"]
        for i in range(0, len(args), 7):
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": args[i : i + 7]}}]
                        },
                    }
                ]
            }
        finish = "tool_calls"
    else:
        for char in message["content"]:
            yield {"choices": [{"index": 0, "delta": {"content": char}}]}
        finish = "stop"
    yield {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    yield {"choices": [], "usage": {"total_tokens": 42}}


class Bytes(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            for index in range(0, len(chunk), 3):
                yield chunk[index : index + 3]


@pytest.mark.parametrize("answer", ['中文😀\n公式 \\(x^2\\) 与 "引号"', "反斜线\\与括号{answer}"])
def test_projection_every_split_decodes_complete_scalars_and_only_root(answer):
    raw = json.dumps({"other": {"answer": "不得展示"}, "topic": "私有备注", "answer": answer})
    prior = ""
    for index in range(len(raw) + 1):
        prefix = answer_prefix(raw[:index])
        assert answer.startswith(prefix) and prefix.startswith(prior)
        assert "不得展示" not in prefix and "私有备注" not in prefix
        assert not any(0xD800 <= ord(c) <= 0xDFFF for c in prefix)
        prior = prefix
    assert prior == answer


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_both_protocols_assemble_split_utf8_tools_usage_and_native_output(protocol, tmp_path):
    settings = Settings(_env_file=None, data_root=tmp_path, protocol=protocol)
    message = deliver("解释😀\n\\(x^2\\)")
    if protocol == "chat_completions":
        chunks = [event(e) for e in chat_events(message)] + [event("[DONE]")]
        native = None
    else:
        function = message["tool_calls"][0]["function"]
        item = {"type": "function_call", "id": "fc-1", "call_id": "call-1", **function}
        native = [{"type": "reasoning", "encrypted_content": "opaque-secret"}, item]
        records = [
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**item, "arguments": ""},
            },
            {"type": "response.reasoning_text.delta", "delta": "reasoning-secret"},
            *[
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 1,
                    "item_id": "fc-1",
                    "delta": c,
                }
                for c in function["arguments"]
            ],
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": native,
                    "usage": {"total_tokens": 42},
                },
            },
        ]
        chunks = [event(e) for e in records]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Bytes(chunks)
        )

    provider = OpenAIProvider(settings, httpx.MockTransport(handler))
    snapshots = []
    result, usage = asyncio.run(
        provider.send_stream(
            provider.build_request([], "system", []),
            lambda text, mode, final=False: snapshots.append((text, mode, final)),
        )
    )
    assert usage == {"total_tokens": 42} and len(requests) == 1 and requests[0]["stream"]
    assert result["tool_calls"] == message["tool_calls"]
    assert snapshots[-1] == ("解释😀\n\\(x^2\\)", "sse", True)
    assert "secret" not in json.dumps(snapshots)
    if native:
        assert result["provider_items"] == native


@pytest.mark.parametrize("kind", ["fallback", "rejected", "truncated", "malformed", "filtered"])
def test_gateway_never_automatically_resends(kind, tmp_path):
    settings = Settings(_env_file=None, data_root=tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        if kind == "fallback":
            return httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "完整回答"}}]}
            )
        if kind == "rejected":
            return httpx.Response(400, json={"error": "private-secret"})
        if kind == "malformed":
            chunks = [event("{broken"), event("[DONE]")]
        elif kind == "filtered":
            chunks = [
                event({"choices": [{"delta": {}, "finish_reason": "content_filter"}]}),
                event("[DONE]"),
            ]
        else:
            chunks = [event({"choices": [{"delta": {"content": "未完成"}}]})]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Bytes(chunks)
        )

    provider = OpenAIProvider(settings, httpx.MockTransport(handler))
    snapshots = []
    request = provider.build_request([], "system", [])
    if kind == "fallback":
        message, _ = asyncio.run(
            provider.send_stream(request, lambda *args: snapshots.append(args))
        )
        assert message["content"] == "完整回答" and snapshots[-1][1] == "json_fallback"
    else:
        with pytest.raises(ModelError) as error:
            asyncio.run(provider.send_stream(request, lambda *args: None))
        assert "private-secret" not in str(error.value)
        assert error.value.outcome == (
            "unknown"
            if kind == "truncated"
            else "failed"
            if kind == "rejected"
            else "returned_invalid"
        )
    assert len(calls) == 1


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_protocol_terminal_closes_connection_without_waiting_for_server_eof(protocol, tmp_path):
    closed = threading.Event()

    class NeverEnds(httpx.AsyncByteStream):
        async def __aiter__(self):
            if protocol == "chat_completions":
                for record in chat_events({"content": "完整讲解"}):
                    yield event(record)
                yield event("[DONE]")
            else:
                yield event(
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [{"type": "output_text", "text": "完整讲解"}],
                                }
                            ],
                        },
                    }
                )
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    settings = Settings(_env_file=None, data_root=tmp_path, protocol=protocol)
    provider = OpenAIProvider(
        settings,
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=NeverEnds()
            )
        ),
    )

    async def check():
        return await asyncio.wait_for(
            provider.send_stream(provider.build_request([], "system", []), lambda *args: None), 0.5
        )

    message, _usage = asyncio.run(check())
    assert message["content"] == "完整讲解" and closed.is_set()


def test_late_second_tool_revokes_projection_instead_of_exposing_arguments():
    assembly = StreamAssembly("chat_completions")
    for record in chat_events(deliver("已经展示的暂存讲解")):
        if record["choices"] and record["choices"][0].get("finish_reason"):
            break
        assembly.feed(record)
    assert assembly.projection() == "已经展示的暂存讲解"
    assembly.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "second",
                                "function": {
                                    "name": "list_materials",
                                    "arguments": '{"private":"secret"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert assembly.projection() == ""


def test_fast_fragments_are_coalesced_before_sqlite_writes(tmp_path):
    answer = "中文😀公式\\(x^2\\)。" * 80
    settings = Settings(_env_file=None, data_root=tmp_path, api_key="test-only", stream_answer=True)
    records = [event(e) for e in chat_events(deliver(answer))] + [event("[DONE]")]
    provider = OpenAIProvider(
        settings,
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Bytes(records)
            )
        ),
    )
    with TestClient(create_app(settings, provider)) as client:
        store = client.app.state.store
        conv = store.conversation(store.space("合并写入")["id"])
        rid = submit(client, conv).json()["run_id"]
        result = ended(client, rid)
        deltas = [e for e in store.progress_events(rid, limit=1000) if e["type"] == "answer.delta"]
        assert result["answer"] == answer
        assert "".join(e["payload"]["text"] for e in deltas) == answer
        assert len(deltas) < len(answer) // 20


class GateStream(httpx.AsyncByteStream):
    def __init__(self, model, message):
        self.model, self.message = model, message

    async def __aiter__(self):
        # Place the answer first so the gate truly blocks after visible decoded text.
        function = self.message["tool_calls"][0]["function"]
        args = json.loads(function["arguments"])
        raw = json.dumps({"answer": args.pop("answer"), **args}, ensure_ascii=False)
        prefix = raw[: raw.index(",")]
        yield event(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "secret-thinking",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "deliver_explanation",
                                        "arguments": prefix,
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        self.model.started.set()
        try:
            while not self.model.release.is_set():
                await asyncio.sleep(0.01)
            yield event(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": raw[len(prefix) :]}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            yield event({"choices": [], "usage": {"total_tokens": 9}})
            yield event("[DONE]")
        except asyncio.CancelledError:
            self.model.cancelled.set()
            raise


class Gateway:
    def __init__(self, messages):
        self.messages, self.calls = messages, []
        self.started, self.release, self.cancelled = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

    def handler(self, request):
        self.calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=GateStream(self, self.messages[len(self.calls) - 1]),
        )


def streaming_client(tmp_path, messages):
    settings = Settings(
        _env_file=None, data_root=tmp_path, api_key="test-only", stream_answer=True, max_steps=3
    )
    gateway = Gateway(messages)
    provider = OpenAIProvider(settings, httpx.MockTransport(gateway.handler))
    client = TestClient(create_app(settings, provider))
    store = client.app.state.store
    conv = store.conversation(store.space("流式学习")["id"])
    return client, gateway, store, conv


def test_draft_refresh_replay_then_atomic_canonical_delivery(tmp_path):
    answer = "近似😀需要条件。\n公式：\\(x^2\\)"
    client, gateway, store, conv = streaming_client(tmp_path, [deliver(answer)])
    rid = submit(client, conv).json()["run_id"]
    snapshot = eventually(lambda: r if (r := store.public_run(rid))["draft"] else None)
    assert snapshot["draft"]["text"] == answer and snapshot["draft"]["offset"] == len(answer)
    assert snapshot["answer"] is None and len(store.messages(conv["id"])) == 1
    assert store.current_learning_state(conv["id"])["revision"] == 0
    assert len(gateway.calls) == 1
    events = client.get(f"/api/runs/{rid}/events?after={snapshot['last_event_seq']}").json()
    assert not any(e["type"] == "answer.delta" for e in events)
    gateway.release.set()
    result = ended(client, rid)
    assert result["answer"] == answer and result["draft"]["status"] == "committed"
    assert (
        len(store.messages(conv["id"])) == 2
        and store.current_learning_state(conv["id"])["revision"] == 1
    )
    replay = client.get(
        f"/api/runs/{rid}/events",
        headers={"Accept": "text/event-stream", "Last-Event-ID": str(snapshot["last_event_seq"])},
    ).text
    assert "run.completed" in replay and "secret-thinking" not in replay
    assert store.progress_events(rid)[-1]["type"] == "run.completed"


def test_cancel_retracts_draft_and_cannot_be_feedback_target(tmp_path):
    client, gateway, store, conv = streaming_client(tmp_path, [deliver("未完成草稿")])
    rid = submit(client, conv).json()["run_id"]
    eventually(lambda: store.public_run(rid)["draft"])
    client.post(f"/api/runs/{rid}/cancel")
    result = ended(client, rid, "cancelled")
    assert gateway.cancelled.is_set() and len(gateway.calls) == 1
    assert result["draft"]["status"] == "revoked"
    assert store.current_learning_state(conv["id"])["revision"] == 0
    assert "未完成草稿" not in json.dumps(store.messages(conv["id"]), ensure_ascii=False)
    response = submit(
        client, conv, identity="request-0002", feedback={"kind": "stuck", "target_run_id": rid}
    )
    assert response.status_code == 422 and len(gateway.calls) == 1


def test_invalid_delivery_retracts_before_agent_corrects_in_new_epoch(tmp_path):
    invalid = deliver("第一份不可提交", approach="invented")
    client, gateway, store, conv = streaming_client(tmp_path, [invalid, deliver("修正后的讲解")])
    rid = submit(client, conv).json()["run_id"]
    eventually(lambda: store.public_run(rid)["draft"])
    gateway.release.set()
    result = ended(client, rid)
    assert result["answer"] == "修正后的讲解" and result["draft"]["epoch"] == 2
    events = store.progress_events(rid)
    assert any(
        e["type"] == "draft.revoked" and e["payload"]["reason"] == "invalid_delivery"
        for e in events
    )
    assert len(gateway.calls) == 2 and store.current_learning_state(conv["id"])["revision"] == 1


@pytest.mark.parametrize("target", ["partial", "returned"])
def test_sigkill_stream_boundary_never_resends_unknown_or_returned_result(tmp_path, target):
    store, settings, _conv, rid = prepared(tmp_path, stream_answer=True, api_key="test-only")
    env = {**os.environ, "RT_ROOT": str(tmp_path), "RT_TARGET": target}

    def spawn():
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("stream_fault_worker.py"))],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    process = spawn()
    try:
        eventually(lambda: (tmp_path / "boundary").exists(), timeout=10)
        old = store.draft_snapshot(rid)
        assert old and old["status"] in {"generating", "validating"}
    finally:
        process.kill()
        process.communicate(timeout=5)
    process = spawn()
    try:
        if target == "returned":
            eventually(lambda: store.public_run(rid)["state"] == "completed", timeout=10)
            result = store.public_run(rid)
            assert result["answer"] == "可恢复的完整讲解😀" and result["recovery_count"] == 1
            # A revoked pre-crash draft stays revoked; the validated canonical answer is separate.
            assert result["draft"]["status"] == "revoked"
        else:
            eventually(lambda: store.public_run(rid)["state"] == "interrupted", timeout=10)
            result = store.public_run(rid)
            assert result["allowed_actions"] == ["cancel", "retry_model"]
            assert result["draft"]["status"] == "revoked" and result["answer"] is None
        assert (tmp_path / "calls").read_text().splitlines() == ["request"]
    finally:
        process.kill()
        process.communicate(timeout=5)
    if target == "partial":
        old_operation = store.operation_at(rid, "main_model:1")["id"]
        gateway = Gateway([deliver("明确重试后的讲解")])
        gateway.release.set()
        with TestClient(
            create_app(settings, OpenAIProvider(settings, httpx.MockTransport(gateway.handler)))
        ) as client:
            run = client.get(f"/api/runs/{rid}").json()
            payload = {
                "action_id": "retry-stream-0001",
                "expected_revision": run["revision"],
                "acknowledge_unknown_request": True,
            }
            response = client.post(f"/api/runs/{rid}/resume", json=payload)
            assert response.status_code == 202
            final = ended(client, rid)
            assert final["draft"]["epoch"] == 2 and final["answer"] == "明确重试后的讲解"
            assert len(gateway.calls) == 1
            owner = store.run(rid)["owner_token"]
            with pytest.raises(RunStopped):
                store.write_draft(rid, owner, old_operation, "迟到的旧草稿")
