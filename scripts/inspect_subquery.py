#!/usr/bin/env python
"""子查询命中诊断：复合问句拆开之后，每个子查询各自能不能找到它那份证据。

上一轮试过"子查询融合 + RRF"，指标被拉低。但 RRF 是全局重排——一路子查询得分高
就会把另一路的证据挤下去，等于没解决"两份证据要同时出现"。真正决定这种做法可行
与否的是更前置的问题：**子查询本身能不能召回它那条证据**。如果连"只让注意力看
滑动窗口 的代价"都召回不到滑动窗口那段，融合策略怎么设计都没用。

本脚本只回答这个问题，不评估指标：

    uv run python scripts/inspect_subquery.py \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
        --ids cross_serve_cost cross_flash_kv cross_long_context cross_kv_reduce
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS, settings_for

SPLIT_POINTS = re.compile(r"[？?]|[；;]|(?<=[^，,])，|以及|各自|分别|对比|相比|和|与|、")


def subqueries(question, min_length=6):
    parts = [part.strip(" ，,。.？?") for part in SPLIT_POINTS.split(question)]
    parts = [part for part in parts if len(part) >= min_length]
    return parts if len(parts) >= 2 else []


def main():
    parser = argparse.ArgumentParser(description="子查询命中诊断")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="local", choices=["local", "ollama", "compatible"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "la-models"))
    args = parser.parse_args()

    from learning_agent.evaluation.graders import matches
    from learning_agent.retrieval import Embedder, Retriever
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="inspect-subquery-"))
    shutil.copy(args.corpus, work / "workspace.db")
    store = Store(work)
    space = store.spaces()[0]["id"]
    materials = store.materials(space)
    assets = {
        key: {"id": next(m["id"] for m in materials if m["name"] == name), "name": name}
        for key, name in DEFAULT_ASSETS.items()
        if any(m["name"] == name for m in materials)
    }
    queries = {q["id"]: q for q in json.loads(args.queries.read_text(encoding="utf-8"))["queries"]}
    settings = settings_for(work, args, rerank_model="")
    retriever = Retriever(store, settings, Embedder(settings))
    retriever.prepare(space)

    for query_id in args.ids:
        query = queries[query_id]
        parts = subqueries(query["question"])
        print(f"\n=== {query_id}  {query['question']}")
        print(f"  子查询（{len(parts)}）：" + " | ".join(parts))
        for part in [query["question"], *parts]:
            hits = retriever.search(space, part, depth=args.depth)["hits"]
            marks = []
            for target in query["expected"]:
                rank = next(
                    (i for i, hit in enumerate(hits, 1) if matches(hit, target, assets)), None
                )
                marks.append(f"{target['asset']}={'第' + str(rank) if rank else '未召回'}")
            print(f"    「{part}」 → " + "，".join(marks))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
