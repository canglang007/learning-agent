"""An actual application process used by execution fault-injection tests (no cloud calls)."""

import asyncio
import json
import os
import time
from pathlib import Path

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.provider import ModelError
from learning_agent.retrieval import Embedder

root = Path(os.environ["RT_ROOT"])
scenario, target = os.environ["RT_SCENARIO"], os.environ.get("RT_TARGET", "")


def record(kind, **values):
    with (root / "calls.jsonl").open("a") as file:
        file.write(json.dumps({"kind": kind, **values}, ensure_ascii=False) + "\n")


def barrier(name):
    if target == name:
        (root / "barrier").write_text(name)
        while True:
            time.sleep(0.02)


def call(name, arguments, identifier):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class Model:
    async def step(self, history, instructions, tools, allow_tools):
        summary = any(t["name"] == "save_context_summary" for t in tools)
        record(
            "summary" if summary else "model",
            tool_ids=[m["tool_call_id"] for m in history if m["role"] == "tool"],
        )
        if summary:
            barrier("summary_sending")
            sources = json.loads(history[-1]["content"])["sources"]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call(
                        "save_context_summary",
                        {
                            "items": [
                                {
                                    "text": "用户在学习近似条件",
                                    "kind": "user_statement",
                                    "user_quote": sources[0]["question"],
                                    "source_run_ids": [sources[0]["run_id"]],
                                }
                            ]
                        },
                        "summary",
                    )
                ],
            }, {"total_tokens": 5}
        barrier("sending")
        if scenario == "network_unknown":
            raise ModelError("请求结果未知", outcome="unknown")
        if any(m["role"] == "tool" for m in history) or scenario == "direct":
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call(
                        "deliver_explanation",
                        {
                            "answer": "近似允许有受控的小误差。",
                            "topic": "近似",
                            "confusion": "",
                            "approach": "example",
                            "reason": "解释近似条件",
                        },
                        "delivery",
                    )
                ],
            }, {"total_tokens": 5}
        material = app.state.store.materials(app.state.store.spaces()[0]["id"])[0]["id"]
        if scenario.startswith("embedding"):
            calls = [call("search_materials", {"query": "近似", "material_id": material}, "search")]
        elif scenario == "image":
            calls = [
                call("inspect_page", {"material_id": material, "page": 1, "region": None}, "image")
            ]
        else:
            calls = [
                call(
                    "read_passage",
                    {"material_id": material, "start_line": i, "end_line": i},
                    f"read-{i}",
                )
                for i in (1, 2)
            ]
        return {"role": "assistant", "content": "", "tool_calls": calls}, {"total_tokens": 5}


class Vectors:
    def __init__(self, settings):
        self.key = Embedder(settings).key

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_async(self, texts):
        record("embedding")
        barrier("embedding_sending")
        return self.embed(texts)


settings = Settings(_env_file=None, **json.loads((root / "settings.json").read_text()))
app = create_app(settings, Model())
if scenario.startswith("embedding"):
    app.state.runtime.retriever.embedder = Vectors(settings)

store = app.state.store
original_checkpoint, original_complete, original_prepare = (
    store.save_checkpoint,
    store.complete_operation,
    store.prepare_operation,
)


def save_checkpoint(*args, **kwargs):
    revision = original_checkpoint(*args, **kwargs)
    phase, payload = args[3], args[4]
    if phase == "tools" and payload["call_index"] == 1:
        barrier("tool_1")
    if phase == "ready_to_commit":
        barrier("ready")
    if phase == "model" and payload["index"] == 1:
        barrier("second_model_prepared")
    return revision


def complete_operation(*args, **kwargs):
    result = original_complete(*args, **kwargs)
    with store.connect() as db:
        purpose = db.execute(
            "SELECT purpose FROM run_operations WHERE id=?", (args[2],)
        ).fetchone()[0]
    if purpose == "main_model":
        barrier("returned")
    elif purpose == "query_embedding":
        barrier("embedding_returned")
    elif purpose == "summary":
        barrier("summary_returned")
    return result


def prepare_operation(*args, **kwargs):
    result = original_prepare(*args, **kwargs)
    if args[2] == "main_model":
        barrier("prepared")
    return result


store.save_checkpoint, store.complete_operation, store.prepare_operation = (
    save_checkpoint,
    complete_operation,
    prepare_operation,
)
original_send = store.send_operation


def send_operation(*args, **kwargs):
    result = original_send(*args, **kwargs)
    with store.connect() as db:
        purpose = db.execute(
            "SELECT purpose FROM run_operations WHERE id=?", (args[2],)
        ).fetchone()[0]
    if purpose == "tool":
        barrier("tool_sending")
    return result


store.send_operation = send_operation
if settings.protocol == "responses":
    original_step = Model.step

    async def native_step(self, *args, **kwargs):
        message, usage = await original_step(self, *args, **kwargs)
        message["provider_items"] = [
            {"id": "opaque-native", "type": "reasoning", "encrypted_content": "private-opaque"},
            *(
                {
                    "type": "function_call",
                    "call_id": c["id"],
                    "name": c["function"]["name"],
                    "arguments": c["function"]["arguments"],
                }
                for c in message.get("tool_calls", [])
            ),
        ]
        return message, usage

    Model.step = native_step


async def main():
    async with app.router.lifespan_context(app):
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
