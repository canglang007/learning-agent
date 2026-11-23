#!/usr/bin/env python
"""精排诊断：目标块在自己材料内排第几，以及"分材料取前 k + 交错合并"后的前 5。

跨讲题（``targets_mode=all``）的失败已经定位到精排，而不是召回池：目标块在材料内
的 BM25+向量名次常在前几名，过一遍 ``bge-reranker-base`` 就掉到 20–50 名。
本脚本做三件只在离线做的事：

1. 把目标所在材料的每一块过精排，报目标名次，并打印排在前面的原文——
   用来判断精排是"排错了"还是"另有道理"；
2. ``--facet-file``：一道题问两件事时，拿整句去和只讲一件事的块逐对打分，精排会偏爱
   关键词覆盖多的块。按材料换成各自的子问句再打分，看名次变不变；
3. ``--merge``：按材料各取前 k 条**交错合并**（A1 B1 A2 B2 …），看前 5 能否同时
   包含两份证据——这是分面召回唯一站得住的合并方式：按全局分数合并会把材料内的
   好名次抹平（材料内第 4 → 全局第 34）。

    uv run python scripts/inspect_ce_neighbors.py \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
        --ids cross_flash_kv --top 3 --merge 3
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS, settings_for


def span_of(chunk):
    if "start_seconds" in chunk:
        return f"{chunk['start_seconds']:.0f}-{chunk['end_seconds']:.0f}s"
    return f"p{chunk.get('page')}"


def main():
    parser = argparse.ArgumentParser(description="目标材料的精排名次与前 N 名原文")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="ollama", choices=["local", "ollama", "compatible"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "la-models"))
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--chars", type=int, default=180)
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="把下一块的文本拼到每块后面再精排（方案 D 的单点验证）",
    )
    parser.add_argument(
        "--facet-file",
        type=Path,
        default=None,
        help="JSON：{query_id: {asset: 该材料对应的子问句}}，逐材料用各自子问句精排",
    )
    parser.add_argument(
        "--query-override",
        default=None,
        help="用这串文本替换问句（例如换成繁体），测精排是否在简繁上失配",
    )
    parser.add_argument(
        "--merge",
        type=int,
        default=0,
        metavar="K",
        help="按材料各取前 K 条交错合并，报告前 5 是否覆盖全部标注目标",
    )
    args = parser.parse_args()

    from learning_agent.evaluation.graders import matches
    from learning_agent.retrieval import CourseIndex, Reranker, sigmoid
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="inspect-ce-"))
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
    settings = settings_for(work, args, rerank_model=args.rerank_model)
    reranker = Reranker(settings)
    facets = json.loads(args.facet_file.read_text(encoding="utf-8")) if args.facet_file else {}
    chunks = CourseIndex(store).chunks(space)

    for query_id in args.ids:
        query = queries[query_id]
        question = args.query_override or query["question"]
        print(f"\n=== {query_id}  {question}")

        ranked = {}  # material_id → 按精排排好的块
        for target in query["expected"]:
            mid = assets[target["asset"]]["id"]
            own = [c for c in chunks if c["id"] == mid]
            if not own:
                print(f"  {target['asset']}：语料里没有")
                continue
            texts = [c["excerpt"] for c in own]
            if args.overlap:
                by_ordinal = {c["ordinal"]: c["excerpt"] for c in own}
                texts = [
                    c["excerpt"] + "\n" + by_ordinal.get(c["ordinal"] + 1, "") for c in own
                ]
            facet_query = facets.get(query_id, {}).get(target["asset"]) or question
            if facet_query != question:
                print(f"  子问句（{target['asset']}）：{facet_query}")
            scores = reranker.scores(facet_query, texts)
            order = sorted(range(len(own)), key=lambda j: (-scores[j], j))
            ranked[mid] = [own[j] for j in order]
            position = next(
                (r for r, j in enumerate(order, 1) if matches(own[j], target, assets)), None
            )
            print(
                f"  目标 {target['asset']} {target['seconds']}s → "
                f"整材料 {len(own)} 块里精排第 {position or '—'}"
            )
            for r, j in enumerate(order[: args.top], 1):
                mark = " ← 目标" if matches(own[j], target, assets) else ""
                text = " ".join(own[j]["excerpt"].split())[: args.chars]
                print(f"    {r}. {span_of(own[j])} p={sigmoid(scores[j]):.3f}{mark}  {text}")
            if position and position > args.top:
                j = order[position - 1]
                text = " ".join(own[j]["excerpt"].split())[: args.chars]
                print(
                    f"    {position}. {span_of(own[j])} p={sigmoid(scores[j]):.3f} ← 目标  {text}"
                )

        if args.merge and ranked:
            queues = {mid: rows[: args.merge] for mid, rows in ranked.items()}
            merged = []
            while any(queues.values()):
                for mid, queue in queues.items():
                    if queue:
                        merged.append((mid, queue.pop(0)))
            top5 = merged[:5]
            covered = [
                next(
                    (r for r, (_, chunk) in enumerate(top5, 1) if matches(chunk, target, assets)),
                    None,
                )
                for target in query["expected"]
            ]
            ok = all(r is not None for r in covered)
            print(
                f"  交错合并（每材料前 {args.merge} 条）→ 前 5 覆盖："
                + ("✓ " if ok else "✗ ")
                + str(covered)
            )
            for r, (mid, chunk) in enumerate(top5, 1):
                name = next((a for a, v in assets.items() if v["id"] == mid), "?")
                print(f"    {r}. {name} {span_of(chunk)}  {' '.join(chunk['excerpt'].split())[:70]}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
