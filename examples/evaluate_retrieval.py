"""Compare unhinted real-course retrieval and optional autonomous learning turns.

Gold positions/rubrics are read only by this runner, never injected into model prompts.
All raw answers, observations, token usage and latency remain under ignored data/.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from learning_agent.agent import run_turn
from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.media import probe
from learning_agent.provider import OpenAIProvider
from learning_agent.references import resolve_references
from learning_agent.retrieval import Retriever
from learning_agent.store import Store
from learning_agent.tools import MaterialTools
from learning_agent.worker import Worker


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def relevant(hit, expected, assets):
    for target in expected:
        if hit["id"] != assets[target["asset"]]["id"]:
            continue
        if "pages" in target and hit.get("page") in target["pages"]:
            return True
        if "seconds" in target and "start_seconds" in hit:
            a, b = target["seconds"]
            if hit["end_seconds"] > a and hit["start_seconds"] < b:
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--agent", action="store_true")
    options = parser.parse_args()
    suite = json.loads(options.suite.read_text())
    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(data_root=root / "workspace")
    store = Store(settings.data_root)
    manifest_path = root / "assets.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"assets": {}}
    with TestClient(create_app(settings)) as client, Worker(settings) as worker:
        if "space_id" not in manifest:
            manifest["space_id"] = client.post(
                "/api/spaces", json={"content": suite["name"]}
            ).json()["id"]
            save(manifest_path, manifest)
        space = manifest["space_id"]
        for key, spec in suite["assets"].items():
            existing = manifest["assets"].get(key)
            if existing and existing.get("preparation_state", "ready") == "ready":
                continue
            path = Path(spec["path"])
            if "source_end_seconds" in spec:
                duration = probe(path)["duration"]
                if (
                    abs(
                        duration
                        - (spec["source_end_seconds"] - spec.get("source_start_seconds", 0))
                    )
                    > 0.5
                ):
                    raise SystemExit("素材时长不完整：" + str(path))
            start = time.monotonic()
            if existing:
                mid = existing["id"]
            else:
                response = client.post(
                    f"/api/spaces/{space}/materials", files={"file": (path.name, path.read_bytes())}
                )
                response.raise_for_status()
                mid = response.json()["id"]
                manifest["assets"][key] = {**spec, "id": mid, "preparation_state": "pending"}
                save(manifest_path, manifest)
            worker.drain()
            base = f"/api/spaces/{space}/materials/{mid}"
            client.put(
                base + "/source",
                json={
                    "url": spec.get("source_url"),
                    "start_seconds": spec.get("source_start_seconds", 0),
                },
            ).raise_for_status()
            if spec.get("transcribe"):
                if store.media(space, mid)["transcript_state"] != "ready":
                    client.post(base + "/transcribe").raise_for_status()
                    worker.drain()
                info = store.media(space, mid)
                if info["transcript_state"] != "ready":
                    raise SystemExit("转写失败：" + str(info.get("transcript_error")))
            elif spec.get("captions"):
                captions = Path(spec["captions"])
                client.post(
                    base + "/captions", files={"file": (captions.name, captions.read_bytes())}
                ).raise_for_status()
            manifest["assets"][key] = {
                **spec,
                "id": mid,
                "preparation_s": round(time.monotonic() - start, 3),
                "preparation_state": "ready",
            }
            save(manifest_path, manifest)
            print(
                json.dumps({"prepared": key, "seconds": manifest["assets"][key]["preparation_s"]}),
                flush=True,
            )
    retriever = Retriever(store, settings)
    start = time.monotonic()
    retriever.prepare(space)
    manifest["index"] = {
        **retriever.status(space),
        "preparation_s": round(time.monotonic() - start, 3),
    }
    save(manifest_path, manifest)
    print(json.dumps({"index": manifest["index"]}, ensure_ascii=False), flush=True)
    if options.prepare_only:
        return
    if not manifest["index"]["dense_ready"]:
        raise SystemExit(
            "向量索引未就绪，不能把降级结果标为RRF实验；请检查assets.json中的index状态。"
        )
    comparisons = []
    for mode in ["lexical_baseline", "bm25", "hybrid"]:
        config = settings.model_copy(update={"retrieval_mode": mode})
        tools = MaterialTools(store, space, retriever=Retriever(store, config))
        rows = []
        for item in suite["queries"]:
            start = time.monotonic()
            result = tools.execute("search_materials", {"query": item["question"]})
            ranks = [
                i
                for i, h in enumerate(result["hits"], 1)
                if relevant(h, item["expected"], manifest["assets"])
            ]
            rows.append(
                {
                    **item,
                    "result": result,
                    "latency_s": round(time.monotonic() - start, 3),
                    "first_relevant_rank": ranks[0] if ranks else None,
                }
            )
        positives = [r for r in rows if r["expected"]]
        comparisons.append(
            {
                "mode": mode,
                "hit_at_5": sum(r["first_relevant_rank"] is not None for r in positives)
                / len(positives),
                "mrr_at_5": sum(
                    1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0
                    for r in positives
                )
                / len(positives),
                "rows": rows,
            }
        )
        print(json.dumps({k: v for k, v in comparisons[-1].items() if k != "rows"}), flush=True)
    save(root / "retrieval.json", comparisons)
    if options.agent:
        asyncio.run(evaluate_agent(suite, store, settings, manifest, root))


async def evaluate_agent(suite, store, settings, manifest, root):
    path = root / "agent.json"
    results = json.loads(path.read_text()) if path.exists() else []
    for mode in ["lexical_baseline", "hybrid"]:
        config = settings.model_copy(update={"retrieval_mode": mode})
        provider = OpenAIProvider(config)
        retriever = Retriever(store, config)
        for case in suite["cases"]:
            if any(r["mode"] == mode and r["id"] == case["id"] for r in results):
                continue
            conv = store.conversation(manifest["space_id"])["id"]
            row = {"mode": mode, "id": case["id"], "conversation_id": conv, "turns": []}
            results.append(row)
            for turn in case["turns"]:
                start = time.monotonic()
                result = await run_turn(
                    store, provider, config, conv, turn["question"], retriever=retriever
                )
                run = store.run(result["run_id"])
                answer, events = store.reference_context(result["run_id"])
                row["turns"].append(
                    {
                        **turn,
                        **result,
                        "latency_s": round(time.monotonic() - start, 3),
                        "events": run["events"],
                        "references": resolve_references(answer, events, manifest["space_id"]),
                        "review": None,
                    }
                )
                save(path, results)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "case": case["id"],
                            "turn": len(row["turns"]),
                            "state": result["state"],
                            "seconds": row["turns"][-1]["latency_s"],
                        }
                    ),
                    flush=True,
                )
                if result["state"] == "failed":
                    break


if __name__ == "__main__":
    main()
