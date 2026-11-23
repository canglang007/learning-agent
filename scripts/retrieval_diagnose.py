#!/usr/bin/env python
"""诊断检索结果的材料集中度：前 5 名是不是被同一份材料占满。

只报 Hit@5 会掩盖一类失败：题目要求两份材料的证据同时进前 5
（``targets_mode=all`` 的跨讲对照题），而候选位被同一讲的高分词块占满，
命中在物理上就不可能发生。这个脚本把每题 top-5 的材料名逐条打出来，
并按材料种数、按 targets_mode 分组统计，用来判断问题出在"没检索到"
还是"检索到了但被同源结果挤掉"。

用法：
    uv run python scripts/retrieval_diagnose.py \
        --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
        --queries evals/retrieval-v5.json --mode hybrid
"""

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS


def short_names(store, space):
    """材料全名 → 标注资产名；不参与评测的材料用截断名，仍能看出是哪一份。"""
    by_name = {name: key for key, name in DEFAULT_ASSETS.items()}
    return {
        material["id"]: by_name.get(material["name"], material["name"][:18])
        for material in store.materials(space)
    }


def main():
    parser = argparse.ArgumentParser(description="检索结果材料集中度诊断")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v5.json"))
    parser.add_argument("--mode", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--rerank-model", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--model-cache", default=str(Path.home() / ".cache" / "la-models")
    )
    args = parser.parse_args()

    from learning_agent.config import Settings
    from learning_agent.evaluation.retrieval import first_rank, modality_of, target_ranks
    from learning_agent.retrieval import Embedder, Retriever
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="retrieval-diag-"))
    shutil.copy(args.corpus, work / "workspace.db")
    store = Store(work)
    space = store.spaces()[0]["id"]
    names = short_names(store, space)
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]

    settings = Settings(
        _env_file=None,
        data_root=work,
        retrieval_mode=args.mode,
        # hybrid 走 config.py 默认的 ollama + bge-m3；bm25 档干脆不开向量。
        embedding_provider="ollama" if args.mode == "hybrid" else "disabled",
        embedding_base_url=args.base_url,
        embedding_cache_dir=args.model_cache,
        rerank_model=args.rerank_model,
    )
    retriever = Retriever(store, settings, Embedder(settings))
    if args.mode == "hybrid":
        retriever.prepare(space)

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

    rows, by_mode = [], {}
    for query in queries:
        if not query.get("expected"):
            continue
        result = retriever.search(space, query["question"])
        hits = result["hits"][:5]
        ranks = target_ranks(hits, query["expected"], assets)
        mode = query.get("targets_mode", "any")
        hit = first_rank(ranks, mode)
        labels = [names.get(hit_item["id"], "?") for hit_item in hits]
        rows.append(
            {
                "id": query["id"],
                "modality": modality_of(query, assets),
                "targets_mode": mode,
                "rank": hit,
                "target_ranks": dict(zip([t["asset"] for t in query["expected"]], ranks)),
                "labels": labels,
                "distinct": len(set(labels)),
            }
        )
        by_mode.setdefault(mode, []).append(rows[-1])

    print(f"\n语料 {args.corpus.name}  题目 {args.queries.name}  模式 {args.mode}"
          f"{' + ' + args.rerank_model if args.rerank_model else ''}\n")
    print(f"{'题目':24s} {'模式':4s} {'首位':>4s}  前5材料")
    for row in rows:
        print(
            f"{row['id']:24s} {row['targets_mode']:4s} {row['rank'] or '—'!s:>4s}  "
            + ",".join(row["labels"])
        )

    print("\n=== 汇总 ===")
    for mode, items in sorted(by_mode.items()):
        hits = sum(item["rank"] is not None for item in items)
        print(f"{mode:4s} 题数 {len(items):2d}  Hit@5 {hits / len(items):.3f}")
    distinct_hist = Counter(row["distinct"] for row in rows)
    print("前5里材料种数分布：" + "，".join(
        f"{n} 种 {c} 题" for n, c in sorted(distinct_hist.items())
    ))
    saturated = [row["id"] for row in rows if row["distinct"] == 1]
    print(f"被单一材料占满前 5：{len(saturated)} 题" + (f"（{'，'.join(saturated)}）" if saturated else ""))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
