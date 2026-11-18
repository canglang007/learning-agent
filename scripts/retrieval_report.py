#!/usr/bin/env python
"""对照 BM25 与混合检索，输出可复现的排名表与分模态指标。

用法：
    uv sync --extra embed
    uv run python scripts/retrieval_report.py --corpus data/workspace.db \
        --queries evals/retrieval-v3.json --assets evals/assets.json

``--corpus`` 是含 chunks 的 workspace.db；``--assets`` 把评测集里的资产名映射到
材料文件名，缺省按内置表猜。报告只衡量"标注目标是否进前 5"，不等于回答正确率。

分模态指标是这份脚本的重点：只报总分时，PDF 命中会把视频/图片证据的缺失盖住。
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_ASSETS = {
    "cs_pdf": "cs336-architecture.pdf",
    "cs_video": "cs336-prenorm.mp4",
    "zh_video": "ntu-tool-use.mp4",
    "optimization": "optimization.pdf",
    "worksheet": "worksheet.pdf",
    "notes": "study-notes.md",
}


def load_assets(path, store, space):
    materials = store.materials(space)
    by_name = {m["name"]: m for m in materials}
    mapping = DEFAULT_ASSETS
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8")).get("assets", {})
        mapping = {
            key: (value["path"].split("/")[-1] if isinstance(value, dict) else value)
            for key, value in raw.items()
        }
    resolved, missing = {}, []
    for key, name in mapping.items():
        if name in by_name:
            material = by_name[name]
            resolved[key] = {
                "id": material["id"],
                "name": name,
                "format": material["format"],
            }
        else:
            missing.append(f"{key}->{name}")
    if missing:
        print(f"警告：以下资产不在语料中，相关题目必然漏检：{', '.join(missing)}", file=sys.stderr)
    return resolved


def run(store, settings, space, queries, assets, embedder=None):
    from learning_agent.evaluation.retrieval import (
        first_rank,
        metrics,
        modality_of,
        target_ranks,
    )
    from learning_agent.retrieval import Retriever

    retriever = Retriever(store, settings, embedder) if embedder else Retriever(store, settings)
    if settings.retrieval_mode == "hybrid":
        retriever.prepare(space)
    rows = []
    for query in queries:
        result = retriever.search(space, query["question"])
        expected = query.get("expected") or []
        mode_of_targets = query.get("targets_mode", "any")
        ranks = target_ranks(result["hits"][:5], expected, assets)
        rows.append(
            {
                "id": query["id"],
                "question": query["question"],
                "modality": modality_of(query, assets),
                "targets_mode": mode_of_targets,
                "annotated": bool(expected),
                "unretrievable": query.get("unretrievable"),
                "first_relevant_rank": first_rank(ranks, mode_of_targets),
                "target_ranks": dict(zip([t["asset"] for t in expected], ranks)),
                "retrieval": result["retrieval"],
                "dense": result["dense_status"],
            }
        )
    scored = [
        row for row in rows if row["annotated"] and not row.get("unretrievable")
    ]
    grouped = {}
    for row in scored:
        grouped.setdefault(row["modality"], []).append(row)
    return {
        "mode": settings.retrieval_mode,
        "hit_at_5": metrics(scored)["hit_at_5"],
        "mrr_at_5": metrics(scored)["mrr_at_5"],
        "by_modality": {name: metrics(items) for name, items in sorted(grouped.items())},
        "rows": rows,
    }


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="检索对照报告")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v3.json"))
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--model", default=None, help="embedding 模型；缺省用配置默认值")
    parser.add_argument("--json", type=Path, help="把完整结果另存为 JSON")
    args = parser.parse_args()

    from learning_agent.config import Settings
    from learning_agent.retrieval import Embedder
    from learning_agent.store import Store

    work = Path(tempfile.mkdtemp(prefix="retrieval-report-"))
    shutil.copy(args.corpus, work / "workspace.db")
    store = Store(work)
    spaces = store.spaces()
    if not spaces:
        raise SystemExit("语料里没有资料空间")
    space = spaces[0]["id"]
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    assets = load_assets(args.assets, store, space)

    base = Settings(_env_file=None, data_root=work, retrieval_mode="bm25")
    results = [run(store, base, space, queries, assets)]

    hybrid = Settings(
        _env_file=None,
        data_root=work,
        retrieval_mode="hybrid",
        embedding_provider="local",
        **({"embedding_model": args.model} if args.model else {}),
    )
    if hybrid.embedding_model:
        embedder = Embedder(hybrid)
        results.append(run(store, hybrid, space, queries, assets, embedder))
    else:
        print("未指定 embedding 模型，跳过混合检索", file=sys.stderr)
    shutil.rmtree(work, ignore_errors=True)

    print(f"\n语料：{args.corpus}  题目：{args.queries}  " f"模型：{hybrid.embedding_model or '-'}")
    print("\n| 模式 | Hit@5 | MRR@5 |\n|---|---|---|")
    for result in results:
        print(f"| {result['mode']} | {fmt(result['hit_at_5'])} | {fmt(result['mrr_at_5'], 3)} |")

    modalities = sorted({row["modality"] for row in results[0]["rows"] if row["annotated"]})
    print("\n分模态（同一条查询只按其主要证据模态归组）：")
    print("| 模态 | 题数 | " + " | ".join(f"{r['mode']} Hit@5" for r in results) + " |")
    print("|---|---|" + "---|" * len(results))
    for name in modalities:
        count = results[0]["by_modality"].get(name, {}).get("queries", 0)
        cells = [fmt(result["by_modality"].get(name, {}).get("hit_at_5")) for result in results]
        print(f"| {name} | {count} | " + " | ".join(cells) + " |")

    print("\n| 题目 | 模态 | " + " | ".join(r["mode"] for r in results) + " |")
    print("|---|---|" + "---|" * len(results))
    for index, row in enumerate(results[0]["rows"]):
        if not row["annotated"]:
            continue
        mark = "（不可文字检索）" if row["unretrievable"] else ""
        cells = []
        for result in results:
            rank = result["rows"][index]["first_relevant_rank"]
            cells.append(f"{rank}" if rank else "—")
        print(f"| {row['id']}{mark} | {row['modality']} | " + " | ".join(cells) + " |")
    if len(results) > 1:
        improved = [
            f"{results[0]['rows'][i]['id']}: {results[0]['rows'][i]['first_relevant_rank'] or '—'}→"
            f"{results[1]['rows'][i]['first_relevant_rank'] or '—'}"
            for i in range(len(results[0]["rows"]))
            if results[0]["rows"][i]["first_relevant_rank"] != results[1]["rows"][i]["first_relevant_rank"]
        ]
        print("\n排名变化：" + ("，".join(improved) if improved else "无"))

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n完整结果已写入 {args.json}")


if __name__ == "__main__":
    main()
