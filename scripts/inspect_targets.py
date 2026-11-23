#!/usr/bin/env python
"""把指定题目的每个标注目标摊开看：候选池里最接近的块排第几、为什么不算命中。

Hit@5 只回答"进没进前五"。跨讲题全灭时，需要分清两种完全不同的失败：
一是目标块根本没被召回（排名靠后或不在池里），二是召回到了但块的区间没有
完整覆盖标注区间（命中判定要求 ``a <= 区间起点`` 且 ``区间终点 <= b``）。
前者要改召回，后者是分块粒度问题，判分口径不能顺手放宽——所以先看事实。

用法：
    uv run python scripts/inspect_targets.py --ids cross_serve_cost cross_flash_kv \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db --depth 120
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS, default_rerank_model, settings_for


def main():
    parser = argparse.ArgumentParser(description="逐目标检查候选位置")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=120)
    parser.add_argument("--rerank-candidates", type=int, default=100)
    parser.add_argument("--rerank-model", default=default_rerank_model())
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="local", choices=["local", "ollama", "compatible"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "la-models"))
    args = parser.parse_args()

    from learning_agent.evaluation.graders import matches
    from learning_agent.retrieval import Embedder, Retriever
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="inspect-targets-"))
    shutil.copy(args.corpus, work / "workspace.db")
    store = Store(work)
    space = store.spaces()[0]["id"]
    materials = store.materials(space)
    assets = {
        key: {
            "id": next(m["id"] for m in materials if m["name"] == name),
            "name": name,
            "format": next(m["format"] for m in materials if m["name"] == name),
        }
        for key, name in DEFAULT_ASSETS.items()
        if any(m["name"] == name for m in materials)
    }
    queries = {q["id"]: q for q in json.loads(args.queries.read_text(encoding="utf-8"))["queries"]}

    settings = settings_for(work, args, rerank_candidates=args.rerank_candidates)
    retriever = Retriever(store, settings, Embedder(settings))
    retriever.prepare(space)

    for query_id in args.ids:
        query = queries[query_id]
        hits = retriever.search(space, query["question"], depth=args.depth)["hits"]
        print(f"\n=== {query_id}  {query['question']}")
        for target in query["expected"]:
            asset = target["asset"]
            mid = assets[asset]["id"]
            rank = next(
                (i for i, hit in enumerate(hits, 1) if matches(hit, target, assets)), None
            )
            print(f"  目标 {asset}  {json.dumps({k: v for k, v in target.items() if k != 'asset'}, ensure_ascii=False)}")
            print(f"    精确命中：{'第 ' + str(rank) + ' 名' if rank else '不在池内（' + str(args.depth) + ' 名以内未覆盖区间）'}")
            same = [(i, hit) for i, hit in enumerate(hits, 1) if hit["id"] == mid]
            if not same:
                print("    该材料在池内没有候选")
                continue
            print(f"    该材料在池内 {len(same)} 条，最好的 5 条：")
            for position, hit in same[:5]:
                span = (
                    f"{hit['start_seconds']:.0f}-{hit['end_seconds']:.0f}s"
                    if "start_seconds" in hit
                    else f"p{hit.get('page')}"
                )
                text = " ".join(hit["excerpt"].split())[:110]
                print(f"      #{position:3d} {span:>12s} {text}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
