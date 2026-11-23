"""rerank 策略对照：hybrid RRF vs 纯 CE 替换 vs 分数空间融合（产品 search 路径，不调用 LLM）。

融合默认权重 rerank_ce_weight=0.9 的实验依据见 docs/retrieval.md；
参数扫描证据见 data/evaluations/multimodal-v1/rerank-alpha-sweep 输出与本脚本 JSON。
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_agent.config import Settings
from learning_agent.evaluation.retrieval import first_rank, metrics, modality_of, target_ranks
from learning_agent.retrieval import Embedder, Retriever
from learning_agent.store import Store

work = Path(tempfile.mkdtemp(prefix="rerank-exp-"))
shutil.copy("data/evaluations/multimodal-v1/corpus/workspace.db", work / "workspace.db")
store = Store(work)
space = store.spaces()[0]["id"]
queries = json.loads(Path("evals/retrieval-v4.json").read_text(encoding="utf-8"))["queries"]

from scripts.retrieval_report import load_assets

assets = load_assets(None, store, space)


def make_settings(rerank_model, ce_weight=0.9):
    return Settings(
        _env_file=None,
        data_root=work,
        retrieval_mode="hybrid",
        embedding_provider="local",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_cache_dir=str(Path.home() / ".cache" / "la-models"),
        rerank_model=rerank_model,
        rerank_ce_weight=ce_weight,
    )


def run(settings):
    retriever = Retriever(store, settings, Embedder(settings))
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
                "modality": modality_of(query, assets),
                "targets_mode": mode_of_targets,
                "annotated": bool(expected),
                "unretrievable": query.get("unretrievable"),
                "first_relevant_rank": first_rank(ranks, mode_of_targets),
                "retrieval": result["retrieval"],
                "rerank_status": result.get("rerank_status"),
            }
        )
    scored = [r for r in rows if r["annotated"] and not r.get("unretrievable")]
    grouped = {}
    for r in scored:
        grouped.setdefault(r["modality"], []).append(r)
    return {
        "mode": settings.rerank_model or "hybrid_rrf",
        "overall": metrics(scored),
        "by_modality": {name: metrics(items) for name, items in sorted(grouped.items())},
        "rows": rows,
    }


runs = {
    "baseline": run(make_settings("")),
    "rerank_replace": run(make_settings("BAAI/bge-reranker-base", ce_weight=1.0)),
    "rerank_fusion_0.9": run(make_settings("BAAI/bge-reranker-base", ce_weight=0.9)),
}
base = {row["id"]: row["first_relevant_rank"] for row in runs["baseline"]["rows"]}
print(f"{'id':24s} {'base':>5s} {'replace':>8s} {'fusion0.9':>10s}")
for rid, b in base.items():
    rep = next(r["first_relevant_rank"] for r in runs["rerank_replace"]["rows"] if r["id"] == rid)
    fus = next(
        r["first_relevant_rank"] for r in runs["rerank_fusion_0.9"]["rows"] if r["id"] == rid
    )
    mark = ""
    if (fus or 99) < (rep or 99):
        mark = "  ← fusion better"
    elif (fus or 99) > (rep or 99):
        mark = "  ← fusion worse"
    print(f"{rid:24s} {b!s:>5s} {rep!s:>8s} {fus!s:>10s}{mark}")
print()
for name, result in runs.items():
    print(
        f"{name:16s} hit@5={result['overall']['hit_at_5']:.3f} "
        f"mrr@5={result['overall']['mrr_at_5']:.3f}"
    )
Path("data/evaluations/multimodal-v1/rerank-fusion.json").write_text(
    json.dumps(runs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
shutil.rmtree(work, ignore_errors=True)
