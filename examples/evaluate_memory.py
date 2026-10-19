"""Real-provider memory acceptance in an isolated workspace; no API keys in reports."""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from learning_agent.agent import run_turn
from learning_agent.config import Settings
from learning_agent.provider import OpenAIProvider
from learning_agent.store import Store


async def evaluate(output):
    if output.exists():
        raise SystemExit("输出目录已存在，请选择新目录，避免覆盖旧结果")
    output.mkdir(parents=True)
    settings = Settings(data_root=output / "workspace", max_steps=6)
    if not settings.api_key.get_secret_value():
        raise SystemExit("请先在本机 .env 配置 LA_API_KEY")
    provider, store = OpenAIProvider(settings), Store(settings.data_root)
    space = store.space("归一化学习")
    material = store.add_material(
        space["id"],
        "notes.md",
        "对非负向量x，若S=sum(x)>0，可令p_i=x_i/S。\n例如x=[2,4]，p=[1/3,2/3]。所有分量除以同一个非零数，任意两项的比值不变；分量和为1。\n这不是减均值除标准差的标准化，也不是L2归一化。",
    )
    topic = store.create_learning_topic(
        space["id"], {"title": "归一化", "material_ids": [material["id"]]}
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model": settings.model,
        "protocol": settings.protocol,
        "cases": [],
    }

    def conversation(selected=None):
        conv = store.conversation(space["id"])
        if selected:
            store.edit_context_settings(
                conv["id"], {"revision": 0, "topic_id": selected["id"], "memory_enabled": True}
            )
        return conv

    async def ask(name, conv, question):
        started = time.monotonic()
        result = await run_turn(store, provider, settings, conv["id"], question)
        record = {
            "name": name,
            "conversation_id": conv["id"],
            "question": question,
            **result,
            "elapsed_s": round(time.monotonic() - started, 3),
            "context": store.run_context(result["run_id"]),
            "events": store.run(result["run_id"])["events"],
        }
        report["cases"].append(record)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {"case": name, "state": result["state"], "answer": result["answer"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if result["state"] == "failed":
            raise SystemExit("真实调用失败，已保存当前报告；不自动重试")
        return result

    source = conversation(topic)
    first = await ask("source", source, "请读取notes.md，解释为什么归一化后比例不变。120字以内。")
    memory = store.create_memory(
        space["id"],
        {
            "topic_id": topic["id"],
            "kind": "preference",
            "text": "讲归一化时，先用2和4算出数值例子，再解释一般符号，不使用水果类比。",
            "material_ids": [],
            "source_run_id": first["run_id"],
        },
    )
    report["memory_id"] = memory["id"]
    await ask("independent_default", conversation(), "请解释归一化后为什么比例不变，120字以内。")
    continued = conversation(topic)
    await ask("continued", continued, "接着讲归一化后为什么比例不变，120字以内。")
    same_name = store.create_learning_topic(
        space["id"], {"title": "归一化", "material_ids": [material["id"]]}
    )
    await ask(
        "same_name_isolated", conversation(same_name), "请解释归一化后为什么比例不变，120字以内。"
    )
    memory = store.edit_memory(
        space["id"],
        memory["id"],
        {"revision": 1, "text": "先给一般公式，再用3和9算一遍；不要沿用2和4作为本轮例子。"},
    )
    await ask("corrected", continued, "继续解释归一化后为什么比例不变，120字以内。")
    memory = store.edit_memory(space["id"], memory["id"], {"revision": 2, "status": "disabled"})
    await ask("disabled", continued, "继续解释归一化后为什么比例不变，120字以内。")
    memory = store.edit_memory(space["id"], memory["id"], {"revision": 3, "status": "active"})
    await ask(
        "current_question_wins",
        continued,
        "本轮不要举例，也不要公式，只给两句直观说明归一化为什么保留比例。",
    )
    store.edit_memory(space["id"], memory["id"], {"revision": 4}, forget=True)
    await ask(
        "forgotten", continued, "我之前保存的讲解顺序偏好是什么？没有可用记录请直说，不要猜。"
    )
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["usage"] = {}
    for case in report["cases"]:
        for event in case["events"]:
            if event["type"] == "model":
                for key, value in event.get("usage", {}).items():
                    if isinstance(value, (int, float)):
                        report["usage"][key] = report["usage"].get(key, 0) + value
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(evaluate(parser.parse_args().output))
