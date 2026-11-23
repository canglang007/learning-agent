#!/usr/bin/env python
"""一次装索引，测三件事：材料多样性配额、交叉编码器重排、两者叠加。

单独跑每个对照都要重新算一遍 942 块的向量，慢而且容易在"哪次用的是哪版语料"
上出错。这里只 prepare 一次，同一批候选上切换后处理策略，指标才可比。

配额的定义（不是为结论服务的规则，而是检索结果该有的性质）：一门课按讲拆成
多份材料后，同一讲的相邻字幕块高度冗余，前 5 名全给同一讲时，读者看不到
"还有哪几处讲过这件事"，跨讲问题也不可能凑齐两份材料的证据。因此限制同一
材料进入前 5 的条数（``--per-material``），不足时按原序回填，单材料语料不受影响。

用法：
    uv run python scripts/retrieval_ablation.py \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db --depth 40
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS, settings_for

# 复合查询的切分点：问句结尾、分句标点、并列与对照连词。
# 「和 / 与 / 各自」这类连词在单主题问句里也会出现，所以切出来的子查询是
# 额外一路召回，原查询始终保留——否则拆错一次就把正确答案挤出去。
SPLIT_POINTS = re.compile(r"[？?]|[；;]|(?<=[^，,])，|以及|各自|分别|对比|相比|和|与|、")


def subqueries(question, min_length=6):
    parts = [part.strip(" ，,。.？?") for part in SPLIT_POINTS.split(question)]
    parts = [part for part in parts if len(part) >= min_length]
    return parts if len(parts) >= 2 else []


def fuse_hit_lists(hit_lists, k=60):
    """按 chunk 位次做 RRF；返回去重后的命中列表（保留首见的那条）。"""
    scores, first = Counter(), {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, 1):
            key = hit["chunk_id"]
            scores[key] += 1 / (k + rank)
            first.setdefault(key, hit)
    return [first[key] for key in sorted(scores, key=lambda key: (-scores[key], key))]


def quota_select(hits, limit, per_material, per_page=2):
    """按顺序准入：同一页 ≤ per_page 条、同一材料 ≤ per_material 条。

    实测结论（v5，942 块）：每材料 3 条把总 Hit@5 从 0.548 压到 0.536，
    2 条压到 0.500，且 4 道跨讲题仍是 0/4——同源冗余不是跨讲题失败的原因，
    另一讲的正确块本来就不在候选池里。因此默认不启用，只用 ``--quota`` 复现。

    两条约束都不硬性丢弃候选：先按约束取，凑不满 limit 时再用剩下的回填，
    这样只有一份材料的语料仍能返回完整条数。
    """
    chosen, deferred, pages, materials = [], [], Counter(), Counter()
    for hit in hits:
        page_key = (hit["id"], hit.get("page") or hit["ordinal"])
        if pages[page_key] >= per_page or materials[hit["id"]] >= per_material:
            deferred.append(hit)
            continue
        pages[page_key] += 1
        materials[hit["id"]] += 1
        chosen.append(hit)
        if len(chosen) == limit:
            return chosen
    for hit in deferred:
        page_key = (hit["id"], hit.get("page") or hit["ordinal"])
        if pages[page_key] >= per_page:
            continue
        pages[page_key] += 1
        chosen.append(hit)
        if len(chosen) == limit:
            break
    return chosen


def evaluate(queries, assets, pick):
    """``pick(query) -> hits``；返回分 targets_mode 的指标与材料种数分布。"""
    from learning_agent.evaluation.retrieval import first_rank, metrics, target_ranks

    rows = []
    for query in queries:
        if not query.get("expected"):
            continue
        mode = query.get("targets_mode", "any")
        hits = pick(query)[:5]
        ranks = target_ranks(hits, query["expected"], assets)
        rows.append(
            {
                "id": query["id"],
                "targets_mode": mode,
                "first_relevant_rank": first_rank(ranks, mode),
                "target_ranks": dict(zip([t["asset"] for t in query["expected"]], ranks)),
                "materials": len({hit["id"] for hit in hits}),
            }
        )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["targets_mode"], []).append(row)
    return {
        "overall": metrics(rows),
        "by_mode": {
            name: {**metrics(items), "ids_missed": [i["id"] for i in items if not i["first_relevant_rank"]]}
            for name, items in sorted(grouped.items())
        },
        "single_material_top5": sum(row["materials"] == 1 for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="检索后处理消融")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--quota", type=int, default=0, help="每材料准入上限；0 关闭（实测降低指标）")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "la-models"))
    parser.add_argument("--model", default=None, help="向量模型；缺省用配置默认（ollama bge-m3）")
    parser.add_argument("--provider", default="ollama", choices=["local", "ollama", "compatible"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from learning_agent.retrieval import Embedder, Retriever
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="retrieval-ablation-"))
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
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]

    def make(model):
        return settings_for(work, args, rerank_model=model, rerank_candidates=args.rerank_candidates)

    plain = Retriever(store, make(""), Embedder(make("")))
    plain.prepare(space)
    print(
        f"语料 {args.corpus}  题目 {args.queries}  候选深度 {args.depth}\n"
        f"向量 {plain.settings.embedding_provider} / {plain.settings.embedding_model}"
        f"  精排 {plain.settings.rerank_model or '关'}",
        flush=True,
    )

    def candidates(query):
        return plain.search(space, query["question"], depth=args.depth)["hits"]

    pool = {query["id"]: candidates(query) for query in queries}
    print("候选池就绪，开始逐配置评估", flush=True)

    configs = {}

    def dump():
        """每算完一组就落盘：中途中断也不至于把前面的对照全丢掉。

        ``meta`` 记死"这组数字是哪个向量模型、哪个精排模型跑出来的"。上一版
        文件里只有分数，模型换了之后没人分得清同一份 JSON 属于哪次实验——
        这也是"用自己换的模型测出来、却当成产品数字"能悄悄发生的原因。
        """
        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "meta": {
                            "corpus": str(args.corpus),
                            "queries": str(args.queries),
                            "provider": plain.settings.embedding_provider,
                            "embedding_model": plain.settings.embedding_model,
                            "rerank_candidates": args.rerank_candidates,
                            "depth": args.depth,
                        },
                        "configs": {
                            name: {k: v for k, v in r.items() if k != "rows"}
                            for name, r in configs.items()
                        },
                    },
                    ensure_ascii=False,
                    indent=1,
                )
                + "\n",
                encoding="utf-8",
            )

    def add(name, result):
        configs[name] = result
        dump()
        print(f"  ✓ {name}: Hit@5 {result['overall']['hit_at_5']:.3f}", flush=True)

    add("hybrid（现状）", evaluate(queries, assets, lambda q: pool[q["id"]]))
    if args.quota:
        add(
            f"hybrid + 每材料{args.quota}条",
            evaluate(queries, assets, lambda q: quota_select(pool[q["id"]], 5, args.quota)),
        )

    # 复合查询：原查询 + 各子查询分别检索，按 chunk 位次融合。
    def multi_query(query, retriever):
        lists = [retriever.search(space, query["question"], depth=args.depth)["hits"]]
        for part in subqueries(query["question"]):
            lists.append(retriever.search(space, part, depth=args.depth)["hits"])
        return fuse_hit_lists(lists)

    multi_pool = {query["id"]: multi_query(query, plain) for query in queries}
    parts_of = {query["id"]: subqueries(query["question"]) for query in queries}
    add("hybrid + 子查询融合", evaluate(queries, assets, lambda q: multi_pool[q["id"]]))
    print(
        "  切出子查询的题："
        + "；".join(f"{qid}={len(parts)}" for qid, parts in parts_of.items() if parts),
        flush=True,
    )

    reranked = Retriever(store, make("BAAI/bge-reranker-base"), Embedder(make("")))
    reranked.prepare(space)
    ce_pool = {}
    started = time.time()
    for index, query in enumerate(queries, 1):
        expected = query.get("expected") or []
        if not expected:
            continue
        ce_pool[query["id"]] = reranked.search(space, query["question"], depth=args.depth)["hits"]
        if index % 20 == 0:
            print(f"  精排 {index}/{len(queries)}  累计 {time.time() - started:.0f}s", flush=True)
    print(f"  精排一组耗时 {time.time() - started:.0f}s", flush=True)
    add("hybrid + CE 重排", evaluate(queries, assets, lambda q: ce_pool[q["id"]]))

    # 子查询融合之后再精排：融合只扩大候选，排序交给 CE。
    from learning_agent.retrieval import fuse_ce_with_rrf

    def multi_then_ce(query):
        fused = multi_pool[query["id"]][: args.rerank_candidates]
        if not fused:
            return []
        scores = reranked.reranker.scores(query["question"], [hit["excerpt"] for hit in fused])
        order = fuse_ce_with_rrf(scores, list(range(len(fused))), 0.9)
        return [fused[j] for j in order] + list(multi_pool[query["id"]][args.rerank_candidates :])

    fused_ce = {}
    for index, query in enumerate(queries, 1):
        if query.get("expected"):
            fused_ce[query["id"]] = multi_then_ce(query)
        if index % 20 == 0:
            print(f"  融合后精排 {index}/{len(queries)}", flush=True)
    add("hybrid + 子查询融合 + CE", evaluate(queries, assets, lambda q: fused_ce[q["id"]]))

    bm25 = Retriever(
        store,
        settings_for(
            work,
            args,
            retrieval_mode="bm25",
            embedding_provider="disabled",
            rerank_model="BAAI/bge-reranker-base",
            rerank_candidates=args.rerank_candidates,
        ),
    )
    add(
        "bm25 + CE 重排（不开向量）",
        evaluate(
            queries, assets, lambda q: bm25.search(space, q["question"], depth=args.depth)["hits"]
        ),
    )

    print("\n| 配置 | 总 Hit@5 | MRR@5 | any Hit@5 | all 命中 | 单一材料占满前5 |")
    print("|---|---|---|---|---|---|")
    for name, result in configs.items():
        any_mode = result["by_mode"].get("any", {})
        all_mode = result["by_mode"].get("all", {})
        all_hits = all_mode.get("queries", 0) - len(all_mode.get("ids_missed", []))
        print(
            f"| {name} | {result['overall']['hit_at_5']:.3f} | {result['overall']['mrr_at_5']:.3f} "
            f"| {any_mode.get('hit_at_5', 0):.3f} | {all_hits}/{all_mode.get('queries', 0)} "
            f"| {result['single_material_top5']} 题 |"
        )

    base = configs["hybrid（现状）"]["rows"]
    for name, result in configs.items():
        if name == "hybrid（现状）":
            continue
        better = [
            f"{after['id']}: {before['first_relevant_rank'] or '—'}→{after['first_relevant_rank']}"
            for before, after in zip(base, result["rows"])
            if before["first_relevant_rank"] != after["first_relevant_rank"]
        ]
        print(f"\n{name} 相对现状的排名变化：" + ("，".join(better) if better else "无"))

    if args.json:
        dump()
        print(f"\n已写入 {args.json}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
