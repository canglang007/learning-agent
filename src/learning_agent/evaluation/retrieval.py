"""Component ranking metrics; an annotated target hit is not answer correctness."""

from collections import Counter
from pathlib import Path
from uuid import uuid4

from learning_agent.evaluation.graders import matches
from learning_agent.evaluation.ledger import Ledger, save_json
from learning_agent.evaluation.runner import configuration
from learning_agent.retrieval import Retriever
from learning_agent.tools import MaterialTools

TARGETS_MODES = {"any", "all"}


def modality_of(query, assets):
    """评测分组标签：显式声明优先，否则按标注资产的格式推断。

    显式声明是必要的：``runtime`` 这类查询同时标注了 PDF 页与视频片段，
    按格式推断只能得到 ``mixed``，而它真正要证明的是"视频证据有没有被取到"。
    """
    if query.get("modality"):
        return query["modality"]
    formats = {
        (assets.get(target.get("asset")) or {}).get("format")
        for target in query.get("expected", [])
    } - {None}
    if len(formats) == 1:
        return formats.pop()
    return "mixed" if formats else "none"


def target_ranks(hits, targets, assets):
    """每个标注目标在 top-k 里第一次出现的位置；未命中为 None。

    不同 target 可以由不同 chunk 满足，这正是多模态查询的常态：
    PDF 页和视频片段不可能落在同一个 chunk 里。
    """
    return [
        next(
            (rank for rank, hit in enumerate(hits, 1) if matches(hit, target, assets)),
            None,
        )
        for target in targets
    ]


def first_rank(ranks, mode):
    """``any``：任一证据到位即可；``all``：全部证据都到位才算命中。

    ``all`` 取最晚到位的那个位置，便于看出"PDF 早就命中、视频拖到第 4 名"。
    """
    if not ranks:
        return None
    if mode == "all":
        return None if any(rank is None for rank in ranks) else max(ranks)
    found = [rank for rank in ranks if rank is not None]
    return min(found) if found else None


def metrics(rows):
    total = len(rows)
    if not total:
        return {"queries": 0, "hit_at_5": None, "mrr_at_5": None}
    hits = sum(row["first_relevant_rank"] is not None for row in rows)
    mrr = sum(
        1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0 for row in rows
    )
    return {"queries": total, "hit_at_5": hits / total, "mrr_at_5": mrr / total}


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
            # 组件评测只报两档召回：精排是叠在上面的第三档，由
            # scripts/rerank_experiment.py 单独测。产品默认开精排，这里显式关掉，
            # 否则"混合"那档会悄悄变成"混合+精排"。
            {"retrieval_mode": mode, "rerank_model": ""},
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
            mode_of_targets = query.get("targets_mode", "any")
            if mode_of_targets not in TARGETS_MODES:
                raise ValueError(f"未知 targets_mode：{mode_of_targets}（{query['id']}）")
            ranks = target_ranks(result["hits"][:5], targets, corpus["assets"])
            rows.append(
                {
                    "id": query["id"],
                    "question": query["question"],
                    "modality": modality_of(query, corpus["assets"]),
                    "targets_mode": mode_of_targets,
                    "target_ranks": {
                        target["asset"]: rank for target, rank in zip(targets, ranks)
                    },
                    "has_annotated_target": bool(targets),
                    "unretrievable": query.get("unretrievable"),
                    "first_relevant_rank": first_rank(ranks, mode_of_targets),
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
        grouped = {}
        for row in scored:
            grouped.setdefault(row["modality"], []).append(row)
        overall = metrics(scored)
        results.append(
            {
                "requested_mode": mode,
                "actual_modes": dict(actual),
                "annotated_target_queries": len(scored),
                "unretrievable_queries": [r["id"] for r in rows if r["unretrievable"]],
                "annotated_hit_at_5": overall["hit_at_5"],
                "annotated_mrr_at_5": overall["mrr_at_5"],
                "by_modality": {
                    name: metrics(items) for name, items in sorted(grouped.items())
                },
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
        "schema_version": 2,
        "kind": "retrieval_component",
        "results": results,
        "note": (
            "非穷尽标注目标Hit/MRR；无标注负例不计召回分母，候选相似不代表模型应回答。"
            "标注目标本身不可经文字检索到达时单列 unretrievable，不算检索失败。"
            "targets_mode=all 要求每个标注目标都被 top-5 中某个条目命中，"
            "用于避免「PDF 命中即算过」掩盖视频/图片证据缺失；by_modality 按证据模态分组。"
            "negative_score_overlap 表示负例最高分不低于命中最低分，说明分数不可用于拒答。"
            "hybrid降级单列，不能声称实际RRF。"
        ),
    }
    save_json(root / "components" / (str(uuid4()) + ".json"), result)
    return result
