"""离线扫描 RRF+CE 融合权重：一次推理收集两种排序，离线扫 alpha 找最优。

产品口径一致地模拟每 (chunk,page) 限 2 条与前 5 截断；
alpha=None 为纯 CE 替换，alpha=0 为纯 RRF（baseline）。
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_agent.config import Settings
from learning_agent.evaluation.retrieval import first_rank, metrics, target_ranks
from learning_agent.retrieval import Embedder, Retriever, bm25, course_scope, fuse
from learning_agent.store import Store

work = Path(tempfile.mkdtemp(prefix="rerank-sweep-"))
shutil.copy("data/evaluations/multimodal-v1/corpus/workspace.db", work / "workspace.db")
store = Store(work)
space = store.spaces()[0]["id"]
queries = json.loads(Path("evals/retrieval-v4.json").read_text(encoding="utf-8"))["queries"]

import math

from scripts.retrieval_report import load_assets


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


assets = load_assets(None, store, space)

settings = Settings(
    _env_file=None,
    data_root=work,
    retrieval_mode="hybrid",
    embedding_provider="local",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embedding_cache_dir=str(Path.home() / ".cache" / "la-models"),
    rerank_model="BAAI/bge-reranker-base",
)
retriever = Retriever(store, settings, Embedder(settings))
retriever.prepare(space)

cases = []
for query in queries:
    chunks = retriever.index.chunks(space, None)
    chunks, search_query, _ = course_scope(query["question"], chunks)
    scores = bm25(search_query, [c["excerpt"] for c in chunks])
    lexical = sorted(
        (i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i)
    )
    vectors = retriever.vectors(chunks)
    if len(vectors) != len(chunks):
        print("skip (no dense):", query["id"])
        continue
    vector = retriever.embedder.embed([search_query])[0]
    sims = [sum(a * b for a, b in zip(vector, vectors[c["chunk_id"]])) for c in chunks]
    dense = sorted(range(len(chunks)), key=lambda i: (-sims[i], i))[:30]
    ranking, _ = fuse(lexical[:30], dense)
    pool = ranking[: settings.rerank_candidates]
    try:
        ce = retriever.reranker.scores(query["question"], [chunks[i]["excerpt"] for i in pool])
    except Exception as error:  # noqa: BLE001
        print("rerank failed:", query["id"], error)
        continue
    ce_order = sorted(range(len(pool)), key=lambda j: (-ce[j], pool[j]))
    ce_prob = {j: _sigmoid(ce[j]) for j in range(len(pool))}
    cases.append(
        {
            "id": query["id"],
            "chunks": chunks,
            "pool": pool,
            "ce_order": ce_order,
            "ce_prob": ce_prob,
            "expected": query.get("expected") or [],
            "targets_mode": query.get("targets_mode", "any"),
        }
    )

K = 60


def fused_order(case, alpha):
    """分数空间融合：alpha*CE概率 + (1-alpha)*归一化RRF分数，返回 pool 位置序列。"""
    pool = case["pool"]
    n = len(pool)
    rrf = {j: 1 / (K + j + 1) for j in range(n)}
    rrf_max = max(rrf.values())
    combined = {
        j: alpha * case["ce_prob"][j] + (1 - alpha) * rrf[j] / rrf_max for j in range(n)
    }
    return sorted(range(n), key=lambda j: (-combined[j], j))


def evaluate(alpha):
    rows = []
    for case in cases:
        pool, ce_order, chunks = case["pool"], case["ce_order"], case["chunks"]
        if alpha is None:
            order = ce_order
        elif alpha == 0:
            order = list(range(len(pool)))
        else:
            order = fused_order(case, alpha)
        hits, per_page = [], {}
        for idx in [pool[j] for j in order]:
            chunk = chunks[idx]
            key = (chunk["id"], chunk.get("page") or chunk["ordinal"])
            per_page[key] = per_page.get(key, 0) + 1
            if per_page[key] > 2:
                continue
            hits.append(chunk)
            if len(hits) == 5:
                break
        ranks = target_ranks(hits, case["expected"], assets)
        rows.append(
            {"id": case["id"], "first_relevant_rank": first_rank(ranks, case["targets_mode"])}
        )
    unretrievable = {q["id"] for q in queries if q.get("unretrievable")}
    annotated = {q["id"] for q in queries if q.get("expected")}
    scored = [r for r in rows if r["id"] in annotated and r["id"] not in unretrievable]
    return metrics(scored), {r["id"]: r["first_relevant_rank"] for r in rows}


alphas = [0.75, 0.8, 0.85, 0.9, 0.95]
base_m, base_rows = evaluate(0)
rep_m, rep_rows = evaluate(None)
results = {a: evaluate(a) for a in alphas}

print(f"{'id':24s} {'base':>5s} {'repl':>5s} " + " ".join(f"a={a:g}".rjust(7) for a in alphas))
for rid, base_value in base_rows.items():
    line = f"{rid:24s} {base_value!s:>5s} {rep_rows[rid]!s:>5s} "
    line += " ".join(str(results[a][1][rid]).rjust(7) for a in alphas)
    print(line)
print()
print(f"{'config':16s} {'hit@5':>7s} {'mrr@5':>7s}")
print(f"{'baseline':16s} {base_m['hit_at_5']:7.3f} {base_m['mrr_at_5']:7.3f}")
print(f"{'replace':16s} {rep_m['hit_at_5']:7.3f} {rep_m['mrr_at_5']:7.3f}")
for a, (m, _) in results.items():
    print(f"fusion a={a:<5g} {m['hit_at_5']:7.3f} {m['mrr_at_5']:7.3f}")

shutil.rmtree(work, ignore_errors=True)
