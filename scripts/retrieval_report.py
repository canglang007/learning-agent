#!/usr/bin/env python
"""对照 BM25 与混合检索，输出可复现的排名表。

用法：
    uv sync --extra embed
    uv run python scripts/retrieval_report.py --corpus data/workspace.db \
        --queries evals/retrieval-v2.json --assets evals/assets.json

``--corpus`` 是含 chunks 的 workspace.db；``--assets`` 把评测集里的资产名映射到
材料文件名，缺省按内置表猜。报告只衡量"标注目标是否进前 5"，不等于回答正确率。
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
    "optimization": "optimization.pdf",
    "worksheet": "worksheet.pdf",
    "notes": "study-notes.md",
}


def load_assets(path, store, space):
    ids = {m["name"]: m["id"] for m in store.materials(space)}
    mapping = DEFAULT_ASSETS
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8")).get("assets", {})
        mapping = {
            key: (value["path"].split("/")[-1] if isinstance(value, dict) else value)
            for key, value in raw.items()
        }
    resolved, missing = {}, []
    for key, name in mapping.items():
        if name in ids:
            resolved[key] = {"id": ids[name]}
        else:
            missing.append(f"{key}->{name}")
    if missing:
        print(f"警告：以下资产不在语料中，相关题目必然漏检：{', '.join(missing)}", file=sys.stderr)
    return resolved


def run(store, settings, space, queries, assets, embedder=None):
    from learning_agent.evaluation.graders import matches
    from learning_agent.retrieval import Retriever

    retriever = Retriever(store, settings, embedder) if embedder else Retriever(store, settings)
    if settings.retrieval_mode == "hybrid":
        retriever.prepare(space)
    rows = []
    for query in queries:
        result = retriever.search(space, query["question"])
        expected = query.get("expected") or []
        rank = next(
            (
                position
                for position, hit in enumerate(result["hits"], 1)
                if expected and any(matches(hit, target, assets) for target in expected)
            ),
            None,
        )
        rows.append(
            {
                "id": query["id"],
                "question": query["question"],
                "annotated": bool(expected),
                "rank": rank,
                "retrieval": result["retrieval"],
                "dense": result["dense_status"],
            }
        )
    scored = [row for row in rows if row["annotated"]]
    hits = sum(1 for row in scored if row["rank"] and row["rank"] <= 5)
    mrr = sum(1 / row["rank"] for row in scored if row["rank"]) / max(1, len(scored))
    return {
        "mode": settings.retrieval_mode,
        "hit_at_5": hits / max(1, len(scored)),
        "mrr_at_5": mrr,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="检索对照报告")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("evals/retrieval-v2.json"))
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

    print(f"\n语料：{args.corpus}  题目：{args.queries}  "
          f"模型：{hybrid.embedding_model or '-'}")
    print("\n| 模式 | Hit@5 | MRR@5 |\n|---|---|---|")
    for result in results:
        print(f"| {result['mode']} | {result['hit_at_5']:.2f} | {result['mrr_at_5']:.3f} |")
    print("\n| 题目 | " + " | ".join(r["mode"] for r in results) + " |")
    print("|---|" + "---|" * len(results))
    for index, row in enumerate(results[0]["rows"]):
        if not row["annotated"]:
            continue
        cells = []
        for result in results:
            rank = result["rows"][index]["rank"]
            cells.append(f"{rank}" if rank else "—")
        print(f"| {row['id']} | " + " | ".join(cells) + " |")
    if len(results) > 1:
        improved = [
            f"{results[0]['rows'][i]['id']}: {results[0]['rows'][i]['rank'] or '—'}→"
            f"{results[1]['rows'][i]['rank'] or '—'}"
            for i in range(len(results[0]["rows"]))
            if results[0]["rows"][i]["rank"] != results[1]["rows"][i]["rank"]
        ]
        print("\n排名变化：" + ("，".join(improved) if improved else "无"))

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n完整结果已写入 {args.json}")


if __name__ == "__main__":
    main()
