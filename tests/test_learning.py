import asyncio
import copy
import json

import httpx
import pytest
from client_helpers import TestClient, close_for
from pydantic import SecretStr

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.provider import ModelError, OpenAIProvider


def call(name, arguments, call_id="call-1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class Scripted:
    def __init__(self, responses):
        self.responses, self.inputs = responses, []

    async def step(self, history, instructions, tools, allow_tools):
        self.inputs.append(copy.deepcopy(history))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {"total_tokens": 10}


def setup(tmp_path, responses):
    model = Scripted(responses)
    settings = Settings(_env_file=None, data_root=tmp_path)
    close_for(tmp_path)
    client = TestClient(create_app(settings, model))
    space = client.post("/api/spaces", json={"content": "学习"}).json()
    material = client.post(
        f"/api/spaces/{space['id']}/materials",
        files={
            "file": (
                "notes.md",
                "近似需要明确条件。\n当相对误差足够小时可以忽略高阶项。".encode(),
                "text/markdown",
            )
        },
    ).json()
    conversation = client.post(f"/api/spaces/{space['id']}/conversations").json()
    return client, model, space, material, conversation


def test_direct_explanation_and_followup_persist_without_forced_retrieval(tmp_path):
    client, model, space, _material, conv = setup(
        tmp_path,
        [
            {"role": "assistant", "content": "先区分近似和相等。"},
            {"role": "assistant", "content": "换个例子：把1.001近似为1。"},
        ],
    )
    url = f"/api/conversations/{conv['id']}/messages"
    first = client.post(url, json={"content": "什么是近似？"}).json()
    assert first["state"] == "completed"
    client.post(url, json={"content": "换个例子"})
    assert model.inputs[1][-2]["content"] == "先区分近似和相等。"
    assert not any(
        e["type"] == "tool" for e in client.get(f"/api/runs/{first['run_id']}").json()["events"]
    )
    close_for(tmp_path)
    reopened = TestClient(create_app(Settings(_env_file=None, data_root=tmp_path), model))
    assert len(reopened.get(url).json()) == 4
    assert len(reopened.get(f"/api/spaces/{space['id']}/materials").json()) == 1


def test_agent_reads_material_then_explains_and_keeps_observation_on_followup(tmp_path):
    client, model, _, material, conv = setup(tmp_path, [])
    model.responses = [
        call("read_passage", {"material_id": material["id"], "start_line": 1, "end_line": 2}),
        {"role": "assistant", "content": "这里省略高阶项的前提是相对误差足够小。"},
        {"role": "assistant", "content": "我们用误差大小来理解。"},
    ]
    url = f"/api/conversations/{conv['id']}/messages"
    result = client.post(url, json={"content": "笔记里为什么能忽略这一项？"}).json()
    assert result["state"] == "completed"
    assert "相对误差" in model.inputs[1][-1]["content"]
    events = client.get(f"/api/runs/{result['run_id']}").json()["events"]
    assert events[1]["name"] == "read_passage"
    assert events[1]["observation"]["start_line"] == 1
    client.post(url, json={"content": "这个条件我没懂"})
    assert any(m["role"] == "tool" for m in model.inputs[2])


def test_no_results_can_lead_to_different_action(tmp_path):
    client, model, _, material, conv = setup(tmp_path, [])
    model.responses = [
        call("search_materials", {"query": "unmatched"}),
        call("list_materials", {}, "call-2"),
        call(
            "read_passage",
            {"material_id": material["id"], "start_line": 1, "end_line": 2},
            "call-3",
        ),
        {"role": "assistant", "content": "找到笔记中的条件了。"},
    ]
    result = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "解释笔记"}
    ).json()
    events = client.get(f"/api/runs/{result['run_id']}").json()["events"]
    tool_events = [e for e in events if e["type"] == "tool"]
    assert tool_events[0]["observation"]["hits"] == []
    assert [e["name"] for e in tool_events] == [
        "search_materials",
        "list_materials",
        "read_passage",
    ]


def test_tools_cannot_read_another_space_and_run_recovers(tmp_path):
    client, model, _, material, _ = setup(tmp_path, [])
    space = client.post("/api/spaces", json={"content": "另一个空间"}).json()
    conv = client.post(f"/api/spaces/{space['id']}/conversations").json()
    model.responses = [
        call("read_passage", {"material_id": material["id"], "start_line": 1, "end_line": 2}),
        {"role": "assistant", "content": "当前空间没有此资料，请选择或上传。"},
    ]
    result = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "解释资料"}
    ).json()
    assert result["state"] == "completed"
    assert json.loads(model.inputs[1][-1]["content"])["error"] == "material_not_found"


def test_missing_key_and_provider_failure_are_explicit(tmp_path):
    settings = Settings(_env_file=None, data_root=tmp_path, api_key=SecretStr(""))
    close_for(tmp_path)
    client = TestClient(create_app(settings))
    space = client.post("/api/spaces", json={"content": "space"}).json()
    conv = client.post(f"/api/spaces/{space['id']}/conversations").json()
    url = f"/api/conversations/{conv['id']}/messages"
    assert client.post(url, json={"content": "hello"}).status_code == 503
    assert client.get(url).json() == []
    close_for(tmp_path)
    client = TestClient(create_app(settings, Scripted([ModelError("模型服务暂不可用")])))
    result = client.post(url, json={"content": "hello"}).json()
    assert result["state"] == "failed"
    assert client.get(f"/api/runs/{result['run_id']}").json()["error"] == "模型服务暂不可用"


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_provider_roundtrip_replays_tools_and_usage(protocol):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if protocol == "chat_completions":
            message = (
                call("list_materials", {})
                if len(requests) == 1
                else {"role": "assistant", "content": "解释"}
            )
            return httpx.Response(
                200, json={"choices": [{"message": message}], "usage": {"total_tokens": 7}}
            )
        output = [
            {
                "type": "function_call",
                "name": "list_materials",
                "arguments": "{}",
                "call_id": "call-1",
            }
        ]
        if len(requests) == 2:
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "解释"}],
                }
            ]
        return httpx.Response(
            200, json={"status": "completed", "output": output, "usage": {"total_tokens": 7}}
        )

    async def exercise():
        settings = Settings(
            _env_file=None, api_key="secret", protocol=protocol, base_url="https://model.test/v1"
        )
        provider = OpenAIProvider(settings, httpx.MockTransport(respond))
        history = [{"role": "user", "content": "q"}]
        message, usage = await provider.step(history, "instructions", [], True)
        history.extend([message, {"role": "tool", "tool_call_id": "call-1", "content": "{}"}])
        final, _ = await provider.step(history, "instructions", [], False)
        assert final["content"] == "解释" and usage["total_tokens"] == 7

    asyncio.run(exercise())
    assert "tools" not in requests[1] and "tool_choice" not in requests[1]
    if protocol == "responses":
        assert requests[1]["store"] is False
        assert requests[1]["input"][-1]["type"] == "function_call_output"
    else:
        assert requests[1]["messages"][-1]["role"] == "tool"


def test_provider_does_not_expose_upstream_body_or_key():
    async def exercise():
        provider = OpenAIProvider(
            Settings(_env_file=None, api_key="private-key"),
            httpx.MockTransport(
                lambda request: httpx.Response(401, text="private-key upstream-body")
            ),
        )
        with pytest.raises(ModelError) as error:
            await provider.step([], "", [], True)
        assert "401" in str(error.value)
        assert "private-key" not in str(error.value)
        assert "upstream-body" not in str(error.value)

    asyncio.run(exercise())


def test_duplicate_tool_call_is_not_reexecuted_and_budget_stops(tmp_path):
    client, model, _, _, conv = setup(tmp_path, [])
    model.responses = [
        call("list_materials", {}),
        call("list_materials", {}, "call-2"),
        {"role": "assistant", "content": "根据已有信息解释，后续还需要确认。"},
    ]
    settings = Settings(_env_file=None, data_root=tmp_path, max_steps=3)
    close_for(tmp_path)
    client = TestClient(create_app(settings, model))
    result = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "解释资料"}
    ).json()
    assert result["state"] == "budget_exhausted"
    assert json.loads(model.inputs[-1][-1]["content"])["error"] == "duplicate_call"
    assert len(model.inputs) == 3


def test_invalid_tool_arguments_are_observations_and_upload_validation(tmp_path):
    client, model, space, _, conv = setup(tmp_path, [])
    model.responses = [
        call("read_passage", {"material_id": "id", "start_line": -1, "end_line": 1000}),
        {"role": "assistant", "content": "需要先明确你指的是哪段。"},
    ]
    result = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "解释这段"}
    ).json()
    assert result["state"] == "completed"
    assert json.loads(model.inputs[-1][-1]["content"])["error"] == "invalid_arguments"
    url = f"/api/spaces/{space['id']}/materials"
    assert client.post(url, files={"file": ("fake.pdf", b"pdf")}).status_code == 422
    assert client.post(url, files={"file": ("empty.txt", b" ")}).status_code == 422


def test_same_conversation_rejects_overlapping_requests(tmp_path):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        class Waiting:
            async def step(self, *args):
                entered.set()
                await release.wait()
                return {"role": "assistant", "content": "解释"}, {}

        app = create_app(Settings(_env_file=None, data_root=tmp_path), Waiting())
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            space = (await client.post("/api/spaces", json={"content": "学习"})).json()
            conv = (await client.post(f"/api/spaces/{space['id']}/conversations")).json()
            url = f"/api/conversations/{conv['id']}/messages"
            task = asyncio.create_task(client.post(url, json={"content": "first"}))
            await entered.wait()
            second = await client.post(url, json={"content": "second"})
            assert second.status_code == 409
            release.set()
            assert (await task).json()["state"] == "completed"
            assert len((await client.get(url)).json()) == 2

    asyncio.run(exercise())
