"""rerank 对照实验：hybrid RRF vs RRF+交叉编码器重排（走产品 search 路径，不调用 LLM）。"""

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


def make_settings(rerank_model):
    return Settings(
        _env_file=None,
        data_root=work,
        retrieval_mode="hybrid",
        embedding_provider="local",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_cache_dir=str(Path.home() / ".cache" / "la-models"),
        rerank_model=rerank_model,
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


baseline = run(make_settings(""))
print(json.dumps({k: v for k, v in baseline.items() if k != "rows"}, ensure_ascii=False))
reranked = run(make_settings("BAAI/bge-reranker-base"))
print(json.dumps({k: v for k, v in reranked.items() if k != "rows"}, ensure_ascii=False))
print("逐题（baseline → rerank）：")
for b, r in zip(baseline["rows"], reranked["rows"]):
    before, after = b["first_relevant_rank"], r["first_relevant_rank"]
    if before == after:
        mark = ""
    elif after is None:
        mark = "  ← 退步（掉出前5）"
    elif before is None:
        mark = "  ← 改善（进前5）"
    elif after < before:
        mark = "  ← 改善"
    else:
        mark = "  ← 退步"
    print(f"{r['id']:24s} {str(before):>5s} → {str(after):>4s}{mark}")
Path("data/evaluations/multimodal-v1/rerank-product-path.json").write_text(
    json.dumps({"baseline": baseline, "reranked": reranked}, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
shutil.rmtree(work, ignore_errors=True)
