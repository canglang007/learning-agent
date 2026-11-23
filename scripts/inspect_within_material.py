#!/usr/bin/env python
"""在单一材料内部排序，看目标块排第几——用来判断"分材料召回"能否救回跨讲题。

跨讲 all 题要求两份材料的关键块同时进前 5。全局排序下它们连候选池都进不去，
但这有两种完全不同的原因：一是目标块在**任何**排序下都排不上（内容与问句确实
不像），二是它只是在全局排序里被别的材料的同类块压住了，而**在它自己那份材料
内部其实排得很前**。后者说明全局排序把跨材料比较和材料内比较混在一起做，
分材料各取 top-k 再合并就能解决；前者只能靠改查询或换模型。

本脚本只报事实，不改排序、不改判分：

    uv run python scripts/inspect_within_material.py \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
        --ids cross_serve_cost cross_flash_kv cross_long_context cross_kv_reduce
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS, settings_for


def main():
    parser = argparse.ArgumentParser(description="材料内部排序诊断")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="local", choices=["local", "ollama", "compatible"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "la-models"))
    args = parser.parse_args()

    from learning_agent.evaluation.graders import matches
    from learning_agent.retrieval import Embedder, Retriever, fuse
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="inspect-within-"))
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

    settings = settings_for(work, args, rerank_model="")
    retriever = Retriever(store, settings, Embedder(settings))
    retriever.prepare(space)
    chunks = retriever.index.chunks(space)

    for query_id in args.ids:
        query = queries[query_id]
        print(f"\n=== {query_id}  {query['question']}")
        question = query["question"]
        vector = retriever.embedder.embed([question])[0]
        vectors = retriever.vectors(chunks)
        for target in query["expected"]:
            asset = target["asset"]
            mid = assets[asset]["id"]
            own = [c for c in chunks if c["id"] == mid]
            if not own:
                print(f"  目标 {asset}：语料里没有这份材料")
                continue
            # 材料内部各算一路并**全量**排序：这里要的是"分材料召回取前 k 条能否
            # 捞到目标"，所以不能像线上那样先把每路截到 30 条再融合——那样算出来的
            # 位次会被截断掩盖，看着像"材料内也找不到"。
            lexical = retriever.lexical_scores(own, question)
            lexical_order = sorted(
                (i for i, score in enumerate(lexical) if score > 0), key=lambda i: (-lexical[i], i)
            )
            dense_order = sorted(
                range(len(own)),
                key=lambda i: (-sum(a * b for a, b in zip(vector, vectors[own[i]["chunk_id"]])), i),
            )
            fused_order, _ = fuse(lexical_order, dense_order)
            lexical_rank = next(
                (r for r, i in enumerate(lexical_order, 1) if matches(own[i], target, assets)), None
            )
            dense_rank = next(
                (r for r, i in enumerate(dense_order, 1) if matches(own[i], target, assets)), None
            )
            fused_rank = next(
                (r for r, i in enumerate(fused_order, 1) if matches(own[i], target, assets)), None
            )
            print(
                f"  目标 {asset}：该材料共 {len(own)} 块 → "
                f"BM25 第 {lexical_rank or '—'} / 向量第 {dense_rank or '—'} / 融合第 {fused_rank or '—'}"
            )
            best = fused_order[: max(3, args.depth)]
            top = [
                i
                for i in best
                if matches(own[i], target, assets)
            ]
            if top:
                hit = own[top[0]]
                span = (
                    f"{hit['start_seconds']:.0f}-{hit['end_seconds']:.0f}s"
                    if "start_seconds" in hit
                    else f"p{hit.get('page')}"
                )
                print(f"    命中块：{span} {' '.join(hit['excerpt'].split())[:100]}")
            else:
                first = own[fused_order[0]]
                span = (
                    f"{first['start_seconds']:.0f}-{first['end_seconds']:.0f}s"
                    if "start_seconds" in first
                    else f"p{first.get('page')}"
                )
                print(f"    材料内第 1 名：{span} {' '.join(first['excerpt'].split())[:100]}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
