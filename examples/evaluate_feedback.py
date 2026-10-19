"""Real-model scripted feedback acceptance using an already prepared course corpus."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from learning_agent.agent import run_turn
from learning_agent.config import Settings
from learning_agent.provider import OpenAIProvider
from learning_agent.references import resolve_references
from learning_agent.retrieval import Retriever
from learning_agent.store import Store
from learning_agent.tools import MaterialTools


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def retrieval_comparison(corpus, output, store, settings, space):
    prior = json.loads((corpus / "retrieval.json").read_text())
    before = next(r for r in prior if r["mode"] == "hybrid")
    tools = MaterialTools(store, space, retriever=Retriever(store, settings))
    manifest = json.loads((corpus / "assets.json").read_text())
    from evaluate_retrieval import relevant

    after = []
    for row in before["rows"]:
        start = time.monotonic()
        result = tools.execute("search_materials", {"query": row["question"]})
        if result.get("retrieval") != "hybrid_rrf":
            raise SystemExit("实际检索已降级，停止RRF对照：" + str(result.get("dense_status")))
        ranks = [
            i
            for i, hit in enumerate(result["hits"], 1)
            if relevant(hit, row["expected"], manifest["assets"])
        ]
        after.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected": row["expected"],
                "before_rank": row["first_relevant_rank"],
                "after_rank": ranks[0] if ranks else None,
                "result": result,
                "latency_s": round(time.monotonic() - start, 3),
            }
        )
    positives = [r for r in after if r["expected"]]
    report = {
        "before_hit_at_5": before["hit_at_5"],
        "after_hit_at_5": sum(r["after_rank"] is not None for r in positives) / len(positives),
        "rows": after,
        "annotation_limit": "预设位置未穷尽所有相关视频，不能视为完整证据召回率。",
    }
    save(output / "retrieval.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False), flush=True
    )


async def evaluate(suite, output, store, settings, space, cases):
    path = output / "report.json"
    report = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "model": settings.model,
            "protocol": settings.protocol,
            "cases": [],
            "review_method": "人工编写反馈，执行Agent待对照资料复核；非真人学习实验",
        }
    )
    model = OpenAIProvider(settings)
    retriever = Retriever(store, settings)
    for case in suite["cases"]:
        if cases and case["id"] not in cases:
            continue
        if any(c["id"] == case["id"] for c in report["cases"]):
            continue
        conv = store.conversation(space)["id"]
        row = {"id": case["id"], "conversation_id": conv, "turns": []}
        report["cases"].append(row)
        for turn in case["turns"]:
            feedback = (
                {
                    "kind": turn["feedback"],
                    "target_run_id": row["turns"][-1]["run_id"],
                    "focus_text": turn.get("focus_text", ""),
                }
                if turn.get("feedback")
                else None
            )
            start = time.monotonic()
            result = await run_turn(
                store,
                model,
                settings,
                conv,
                turn["question"],
                retriever=retriever,
                feedback=feedback,
            )
            run = store.run(result["run_id"])
            answer, events = store.reference_context(result["run_id"])
            row["turns"].append(
                {
                    **turn,
                    **result,
                    "feedback_payload": feedback,
                    "latency_s": round(time.monotonic() - start, 3),
                    "answer_characters": len(result["answer"]),
                    "events": run["events"],
                    "learning_state": store.learning_state(conv),
                    "references": resolve_references(answer, events, space),
                    "review": None,
                }
            )
            save(path, report)
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "turn": len(row["turns"]),
                        "state": result["state"],
                        "characters": len(result["answer"]),
                        "seconds": row["turns"][-1]["latency_s"],
                        "tools": [e["name"] for e in run["events"] if e["type"] == "tool"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result["state"] != "completed":
                break
    turns = [t for c in report["cases"] for t in c["turns"]]
    save(
        output / "summary.json",
        {
            "turns": len(turns),
            "completed": sum(t["state"] == "completed" for t in turns),
            "median_characters": statistics.median(t["answer_characters"] for t in turns)
            if turns
            else None,
            "note": "长度与执行状态不替代误解纠正和学习效果复核。",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path, required=True, help="evaluate_retrieval.py已准备的输出目录"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=Path("examples/suites/learning_feedback.json")
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "assets.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings(data_root=args.corpus / "workspace", retrieval_mode="hybrid")
    store = Store(settings.data_root)
    retrieval_comparison(args.corpus, args.output, store, settings, manifest["space_id"])
    if not args.retrieval_only:
        if not settings.api_key.get_secret_value():
            raise SystemExit("请先配置LA_API_KEY。")
        asyncio.run(
            evaluate(
                json.loads(args.suite.read_text()),
                args.output,
                store,
                settings,
                manifest["space_id"],
                args.cases,
            )
        )


if __name__ == "__main__":
    main()
