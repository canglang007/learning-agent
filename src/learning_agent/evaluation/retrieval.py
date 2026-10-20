"""Component ranking metrics; an annotated target hit is not answer correctness."""

from collections import Counter
from pathlib import Path
from uuid import uuid4

from learning_agent.evaluation.graders import matches
from learning_agent.evaluation.ledger import Ledger, save_json
from learning_agent.evaluation.runner import configuration
from learning_agent.retrieval import Retriever
from learning_agent.tools import MaterialTools


def evaluate_retrieval(experiment, queries, settings, *, max_embeddings=12):
    root = Path(experiment)
    ledger = Ledger(root / "experiment.db")
    corpus = ledger.get("corpus")
    if not corpus:
        raise ValueError("需要已有准备完成的资料基线")
    limits = ledger.get("limits")
    limits.setdefault("component_query_embedding", max_embeddings)
    ledger.put("limits", limits)
    results = []
    for mode in ("bm25", "hybrid"):
        config = configuration(
            settings,
            {"retrieval_mode": mode},
            data_root=root / "corpus",
            evaluation_budget_path=None,
        )
        from learning_agent.store import Store

        store = Store(config.data_root)
        retriever = Retriever(store, config)
        tools = MaterialTools(store, corpus["space_id"], retriever=retriever)
        rows = []
        for query in queries:
            if mode == "hybrid" and retriever.status(corpus["space_id"])["dense_ready"]:
                ledger.reserve(str(uuid4()), "component_query_embedding")
            result = tools.execute("search_materials", {"query": query["question"]})
            targets = query.get("expected", [])
            ranks = [
                rank
                for rank, hit in enumerate(result["hits"][:5], 1)
                if any(matches(hit, target, corpus["assets"]) for target in targets)
            ]
            rows.append(
                {
                    "id": query["id"],
                    "question": query["question"],
                    "has_annotated_target": bool(targets),
                    "first_relevant_rank": ranks[0] if ranks else None,
                    "actual_retrieval": result.get("retrieval"),
                    "dense_status": result.get("dense_status"),
                    "hits": result["hits"],
                }
            )
        positives = [r for r in rows if r["has_annotated_target"]]
        actual = Counter(r["actual_retrieval"] for r in rows)
        results.append(
            {
                "requested_mode": mode,
                "actual_modes": dict(actual),
                "annotated_target_queries": len(positives),
                "annotated_hit_at_5": sum(r["first_relevant_rank"] is not None for r in positives)
                / len(positives)
                if positives
                else None,
                "annotated_mrr_at_5": sum(
                    1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0
                    for r in positives
                )
                / len(positives)
                if positives
                else None,
                "rows": rows,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "retrieval_component",
        "results": results,
        "note": "非穷尽标注目标Hit/MRR；无标注负例不计召回分母，候选相似不代表模型应回答。hybrid降级单列，不能声称实际RRF。",
    }
    save_json(root / "components" / (str(uuid4()) + ".json"), result)
    return result
