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
                    "unretrievable": query.get("unretrievable"),
                    "first_relevant_rank": ranks[0] if ranks else None,
                    "actual_retrieval": result.get("retrieval"),
                    "dense_status": result.get("dense_status"),
                    "hits": result["hits"],
                }
            )
        # 标注目标不可经文字检索到达的查询单列，不进召回分母，也不冒充检索失败。
        scored = [r for r in rows if r["has_annotated_target"] and not r["unretrievable"]]
        negatives = [r for r in rows if not r["has_annotated_target"]]
        actual = Counter(r["actual_retrieval"] for r in rows)
        hit_scores = [
            r["hits"][0]["score"]
            for r in scored
            if r["first_relevant_rank"] is not None and r["hits"] and "score" in r["hits"][0]
        ]
        negative_scores = [r["hits"][0]["score"] for r in negatives if r["hits"]]
        results.append(
            {
                "requested_mode": mode,
                "actual_modes": dict(actual),
                "annotated_target_queries": len(scored),
                "unretrievable_queries": [r["id"] for r in rows if r["unretrievable"]],
                "annotated_hit_at_5": sum(r["first_relevant_rank"] is not None for r in scored)
                / len(scored)
                if scored
                else None,
                "annotated_mrr_at_5": sum(
                    1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0 for r in scored
                )
                / len(scored)
                if scored
                else None,
                "negative_queries": len(negatives),
                "negative_top1_scores": [round(s, 4) for s in negative_scores],
                "negative_score_overlap": (
                    bool(hit_scores and negative_scores and max(negative_scores) >= min(hit_scores))
                    if hit_scores and negative_scores
                    else None
                ),
                "rows": rows,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "retrieval_component",
        "results": results,
        "note": (
            "非穷尽标注目标Hit/MRR；无标注负例不计召回分母，候选相似不代表模型应回答。"
            "标注目标本身不可经文字检索到达时单列 unretrievable，不算检索失败。"
            "negative_score_overlap 表示负例最高分不低于命中最低分，说明分数不可用于拒答。"
            "hybrid降级单列，不能声称实际RRF。"
        ),
    }
    save_json(root / "components" / (str(uuid4()) + ".json"), result)
    return result
