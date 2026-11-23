#!/usr/bin/env python
"""离线验证"分材料召回 + 精排"能不能救回跨讲题（不改产品代码）。

背景见 ``docs/cross-material-retrieval-plan.md``：4 道 ``targets_mode=all`` 的题
要求两份材料的证据同时进前 5，而全局排序下它们连候选池都进不去。原因不是
"被别的材料挤掉"，是目标块在**自己的材料里**也排不到前面（hy_flash 那块排 67/69）。
所以这里做两件只在离线做的事：

1. **分面召回**：每份材料各取前 k 条（材料内 BM25+向量 RRF），合并成一个池子；
2. **整池精排**：把池子里所有候选一次性交给交叉编码器，按产品的
   ``combined_scores(weight=0.9)`` 重新排序，看两份证据能不能同时进前 5。

同时报"门控版"：只用全局排序前 N 名里出现过的材料做分面召回——这是线上真能
接受的代价形态（90 条候选 ≈ 4 秒），而全库 11 份材料 × 30 = 330 条 ≈ 15 秒。

    uv run python scripts/inspect_facet_recall.py \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
        --ids cross_serve_cost cross_flash_kv cross_long_context cross_kv_reduce

只报事实，不改判分：命中判定仍走 ``evaluation.graders.matches``，区间仍是标注原文。
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS, settings_for


def fused_within(retriever, own, question, vector, vectors, cap=30):
    """材料内 BM25+向量 RRF，返回材料内顺序（局部下标）。

    每路先截到 ``cap`` 条再融合，与 ``Retriever.search`` 的做法一致——
    否则离线口径比线上宽，结论会偏乐观。
    """
    lexical = retriever.lexical_scores(own, question)
    lexical_order = sorted(
        (i for i, score in enumerate(lexical) if score > 0), key=lambda i: (-lexical[i], i)
    )[:cap]
    dense_order = sorted(
        range(len(own)),
        key=lambda i: (-sum(a * b for a, b in zip(vector, vectors[own[i]["chunk_id"]])), i),
    )[:cap]
    from learning_agent.retrieval import fuse

    order, _ = fuse(lexical_order, dense_order)
    return order


def rank_of(order, chunks, target, assets, matches):
    return next(
        (r for r, i in enumerate(order, 1) if matches(chunks[i], target, assets)), None
    )


def span_of(chunk):
    if "start_seconds" in chunk:
        return f"{chunk['start_seconds']:.0f}-{chunk['end_seconds']:.0f}s"
    return f"p{chunk.get('page')}"


def main():
    parser = argparse.ArgumentParser(description="分材料召回 + 精排的离线验证")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[5, 10, 20, 30, 40, 60], help="每材料取前 k 条"
    )
    parser.add_argument(
        "--gate-ns", nargs="+", type=int, default=[10, 20], help="门控：全局前 N 名涉及的材料"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="ollama", choices=["local", "ollama", "compatible"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "la-models"))
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--ce-weight", type=float, default=0.9)
    parser.add_argument(
        "--ce-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.5, 0.7, 0.9],
        help="精排融合权重扫描；0 为纯 RRF 位次、1 为纯 CE",
    )
    parser.add_argument(
        "--sweep-k", type=int, default=30, help="权重扫描用的每材料取 k"
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from learning_agent.evaluation.graders import matches
    from learning_agent.retrieval import Embedder, Retriever, combined_scores, fuse
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="inspect-facet-"))
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
    retriever = Retriever(store, settings, Embedder(settings))
    retriever.prepare(space)
    chunks = retriever.index.chunks(space)
    vectors = retriever.vectors(chunks)
    print(f"语料 {len(chunks)} 块 / {len(materials)} 份材料；模型 {settings.embedding_model}")
    print(f"精排 {settings.rerank_model}；融合权重 {args.ce_weight}")

    # 全局下标 → 材料分组的全局下标
    by_material = {}
    for i, chunk in enumerate(chunks):
        by_material.setdefault(chunk["id"], []).append(i)

    report = []
    for query_id in args.ids:
        query = queries[query_id]
        question = query["question"]
        targets = query["expected"]
        print(f"\n=== {query_id}  {question}")

        vector = retriever.embedder.embed([question])[0]

        # 全局排序（产品口径），用来给出"门控会选中哪些材料"。
        global_result = retriever.search(space, question, depth=max(args.gate_ns))
        global_ids = [hit["id"] for hit in global_result["hits"]]
        gate_materials = {
            n: {mid for mid in global_ids[:n]} for n in args.gate_ns
        }
        print(
            "  全局前 "
            f"{max(args.gate_ns)} 名涉及材料数："
            + "，".join(f"前{n}={len(s)}份" for n, s in gate_materials.items())
        )

        # 每份材料内的完整融合顺序（只算一次，k 扫描复用前缀）。
        within = {}
        for mid, idx in by_material.items():
            own = [chunks[i] for i in idx]
            local = fused_within(retriever, own, question, vector, vectors)
            within[mid] = [idx[j] for j in local]

        k_max = max(args.ks)
        pool_max, _ = fuse(*[order[:k_max] for order in within.values() if order])
        started = time.perf_counter()
        ce = retriever.reranker.scores(question, [chunks[i]["excerpt"] for i in pool_max])
        ce_seconds = time.perf_counter() - started
        ce_by_chunk = {i: score for i, score in zip(pool_max, ce)}
        print(
            f"  全库分面池 k={k_max}：{len(pool_max)} 条，整池精排 {ce_seconds:.1f} 秒"
            f"（{ce_seconds / max(1, len(pool_max)) * 20:.1f} 秒/20 条）"
        )

        # 闭包把当前查询的 within/ce_by_chunk/targets 绑成默认参数：循环下一轮会覆盖
        # 同名变量，不绑就会拿错查询的分数（B023）。
        def evaluate(material_ids, k, weight=None, within=within, ce_by_chunk=ce_by_chunk, targets=targets):
            weight = args.ce_weight if weight is None else weight
            lists = [within[mid][:k] for mid in material_ids if within.get(mid)]
            if not lists:
                return None
            pool, _ = fuse(*lists)
            values = combined_scores([ce_by_chunk[i] for i in pool], pool, weight)
            order = sorted(range(len(pool)), key=lambda j: (-values[j], pool[j]))
            final = [pool[j] for j in order]
            ranks = [rank_of(final, chunks, t, assets, matches) for t in targets]
            pure_ce = sorted(pool, key=lambda i: (-ce_by_chunk[i], i))
            ce_ranks = [rank_of(pure_ce, chunks, t, assets, matches) for t in targets]
            ok = all(r is not None and r <= 5 for r in ranks)
            return {
                "k": k,
                "materials": len(lists),
                "pool": len(pool),
                "ranks": ranks,
                "ce_ranks": ce_ranks,
                "hit": ok,
                "top": [
                    {
                        "asset": next(
                            (a for a, v in assets.items() if v["id"] == chunks[i]["id"]), "?"
                        ),
                        "span": span_of(chunks[i]),
                        "score": round(values[pool.index(i)], 4),
                    }
                    for i in final[:5]
                ],
            }

        row = {"id": query_id, "question": question, "full": {}, "gated": {}}
        print(
            "  全库分面（不改门控）："
            + "  ".join(
                f"k={k}:"
                + ("✓" if (r := evaluate(by_material.keys(), k))["hit"] else "✗")
                + f"{r['ranks']}"
                for k in args.ks
            )
        )
        row["full"] = {str(k): evaluate(by_material.keys(), k) for k in args.ks}
        for n in args.gate_ns:
            print(
                f"  门控前{n}（{len(gate_materials[n])} 份材料）："
                + "  ".join(
                    f"k={k}:"
                    + ("✓" if (r := evaluate(gate_materials[n], k))["hit"] else "✗")
                    + f"{r['ranks']}"
                    for k in args.ks
                )
            )
            row["gated"][str(n)] = {str(k): evaluate(gate_materials[n], k) for k in args.ks}
        row["weight_sweep"] = {
            str(w): {
                "full": evaluate(by_material.keys(), args.sweep_k, w),
                "gated20": evaluate(gate_materials[max(args.gate_ns)], args.sweep_k, w),
            }
            for w in args.ce_weights
        }

        best = row["full"][str(max(args.ks))]
        if best:
            print("  k 最大时前 5：")
            for item in best["top"]:
                print(f"    {item['score']:.3f}  {item['asset']} {item['span']}")
            print(f"  目标名次（CE 纯序）：{best['ce_ranks']}")

        # CE 只看"这段和问句是否匹配"，不看别的材料；所以它在目标自己材料内部的名次，
        # 能区分两种完全不同的失败：CE 认可这块、只是被跨材料融合挤下去（结构问题），
        # 还是 CE 本身就不认（模型问题）。
        print("  目标在自己材料内：RRF 名次 / 整材料过 CE 的名次")
        ce_within = {}
        for target, rank_final in zip(targets, best["ranks"] if best else []):
            mid = assets[target["asset"]]["id"]
            idx = by_material.get(mid, [])
            own = [chunks[i] for i in idx]
            rrf_rank = rank_of(within[mid], chunks, target, assets, matches) if idx else None
            ce_ranks_local = retriever.reranker.scores(question, [c["excerpt"] for c in own])
            ce_order = [
                idx[j] for j in sorted(range(len(own)), key=lambda j: (-ce_ranks_local[j], j))
            ]
            ce_rank = rank_of(ce_order, chunks, target, assets, matches)
            ce_within[target["asset"]] = {"rrf": rrf_rank, "ce": ce_rank, "of": len(own)}
            print(
                f"    {target['asset']}（{len(own)} 块）：RRF 第 {rrf_rank or '—'} / "
                f"CE 第 {ce_rank or '—'}　最终第 {rank_final or '—'}"
            )
        row["ce_within"] = ce_within
        report.append(row)

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {args.json}")

    print(f"\n## 精排权重扫描（每材料 k={args.sweep_k}，括号为两份证据的名次）")
    print("| 权重 | 模式 | " + " | ".join(row["id"] for row in report) + " |")
    print("|---|---|" + "---|" * len(report))
    for weight in args.ce_weights:
        for scope in ("full", "gated20"):
            cells = []
            for row in report:
                result = row["weight_sweep"][str(weight)][scope]
                cells.append(
                    ("✓ " if result["hit"] else "✗ ") + str(result["ranks"])
                    if result
                    else "—"
                )
            print(f"| {weight} | {scope} | " + " | ".join(cells) + " |")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
