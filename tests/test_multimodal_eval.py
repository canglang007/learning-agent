"""多模态评测：分模态指标、强制命中语义与图像证据门槛。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hit(material_id, **extra):
    return {"material_id": material_id, **extra}


def test_modality_label_prefers_explicit_declaration():
    from learning_agent.evaluation.retrieval import modality_of

    assets = {"pdf": {"id": "p", "format": "pdf"}, "vid": {"id": "v", "format": "video"}}
    assert modality_of({"modality": "video", "expected": [{"asset": "pdf"}]}, assets) == "video"
    assert modality_of({"expected": [{"asset": "pdf"}]}, assets) == "pdf"
    assert modality_of({"expected": [{"asset": "pdf"}, {"asset": "vid"}]}, assets) == "mixed"
    assert modality_of({"expected": []}, assets) == "none"


def test_any_mode_keeps_legacy_single_hit_semantics():
    from learning_agent.evaluation.retrieval import first_rank, target_ranks

    assets = {"pdf": {"id": "p"}, "vid": {"id": "v"}}
    targets = [{"asset": "pdf", "pages": [16]}, {"asset": "vid", "seconds": [70, 82]}]
    hits = [hit("p", page=16)]
    ranks = target_ranks(hits, targets, assets)
    assert ranks == [1, None]
    assert first_rank(ranks, "any") == 1


def test_all_mode_fails_when_video_evidence_is_missing():
    from learning_agent.evaluation.retrieval import first_rank, target_ranks

    assets = {"pdf": {"id": "p"}, "vid": {"id": "v"}}
    targets = [{"asset": "pdf", "pages": [16]}, {"asset": "vid", "seconds": [70, 82]}]
    only_pdf = [hit("p", page=16)]
    assert first_rank(target_ranks(only_pdf, targets, assets), "all") is None

    both = [hit("p", page=16), hit("v", start_seconds=45, end_seconds=89)]
    assert first_rank(target_ranks(both, targets, assets), "all") == 2


def test_all_mode_reports_the_latest_evidence_position():
    from learning_agent.evaluation.retrieval import first_rank, target_ranks

    assets = {"pdf": {"id": "p"}, "vid": {"id": "v"}}
    targets = [{"asset": "pdf", "pages": [16]}, {"asset": "vid", "seconds": [70, 82]}]
    hits = [
        hit("v", start_seconds=45, end_seconds=89),
        hit("p", page=16),
    ]
    assert first_rank(target_ranks(hits, targets, assets), "all") == 2


def test_metrics_group_by_modality():
    from learning_agent.evaluation.retrieval import metrics

    rows = [
        {"first_relevant_rank": 1, "modality": "video"},
        {"first_relevant_rank": None, "modality": "video"},
        {"first_relevant_rank": 2, "modality": "pdf"},
    ]
    assert metrics(rows[:2]) == {"queries": 2, "hit_at_5": 0.5, "mrr_at_5": 0.5}
    assert metrics([])["hit_at_5"] is None


def test_silent_video_answer_without_frames_fails():
    from learning_agent.evaluation.graders import grade

    trace = {
        "state": "completed",
        "answer": "这段讲的是 pre-norm 与 post-norm 的对比。",
        "messages": [{"role": "assistant", "kind": "answer", "content": "解释"}],
        "tools": [{"name": "inspect_media", "observation": {"material_id": "sv"}}],
        "operations": [
            {
                "id": "op",
                "purpose": "main_model",
                "input_ref": {"schema_version": 1, "images": [], "evidence": []},
            }
        ],
    }
    gold = {
        "evidence_targets": [
            {"asset": "sv", "seconds": [5, 25], "modality": "image"},
        ]
    }
    assets = {"sv": {"id": "sv"}}
    assert grade(trace, gold, assets)["checks"][1]["status"] == "fail"

    trace["operations"][0]["input_ref"]["images"] = [{"material_id": "sv", "seconds": 12}]
    assert grade(trace, gold, assets)["checks"][1]["status"] == "pass"


def test_chart_reading_requires_the_right_page_image():
    from learning_agent.evaluation.graders import grade

    gold = {"evidence_targets": [{"asset": "opt", "pages": [51], "modality": "image"}]}
    trace = {
        "state": "completed",
        "answer": "条件数问题。",
        "messages": [{"role": "assistant", "kind": "answer", "content": "解释"}],
        "tools": [],
        "operations": [
            {
                "id": "op",
                "purpose": "main_model",
                "input_ref": {
                    "schema_version": 1,
                    "images": [{"asset_path": "materials/opt/page-51-full.png"}],
                    "evidence": [],
                },
            }
        ],
    }
    assert grade(trace, gold, {"opt": {"id": "opt"}})["checks"][1]["status"] == "pass"
    trace["operations"][0]["input_ref"]["images"] = [
        {"asset_path": "materials/opt/page-50-full.png"}
    ]
    assert grade(trace, gold, {"opt": {"id": "opt"}})["checks"][1]["status"] == "fail"


def test_retrieval_v3_targets_stay_inside_asset_time_ranges():
    """防止再出现超出视频时长的永假锚点（v2 的 859 秒就是这么漏掉的）。"""

    suite = json.loads((ROOT / "evals" / "retrieval-v3.json").read_text(encoding="utf-8"))
    durations = {"cs_video": 198.0, "zh_video": 270.0}
    for query in suite["queries"]:
        assert query.get("targets_mode", "any") in {"any", "all"}
        for target in query.get("expected", []):
            seconds = target.get("seconds")
            anchors = target.get("anchors") or []
            points = list(seconds or []) + anchors
            if not points:
                continue
            limit = durations.get(target["asset"])
            if limit is None:
                continue
            assert all(0 <= point <= limit for point in points), (
                query["id"],
                target["asset"],
                points,
            )


def test_learning_core_v2_gives_multimodal_cases_real_targets():
    suite = json.loads((ROOT / "evals" / "learning-core-v4.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in suite["cases"]}
    for name in ("silent_video", "zh_chart"):
        gold = cases[name]["turns"][0]["gold"]
        assert gold["evidence_targets"], f"{name} 缺少证据锚点"
        assert gold["evidence_targets"][0]["modality"] == "image"

    silent = cases["silent_video"]["turns"][0]
    assert silent["input"]["focus"]["asset"] == "silent_video"
    assert silent["gold"]["evidence_targets"][0]["seconds"] == [5, 25]

    chart = cases["zh_chart"]["turns"][0]
    assert chart["input"]["focus"]["page"] == 51

    assert cases["video_limit"]["turns"][0]["gold"]["forbidden_claims"]


def test_learning_core_v2_suite_loads_through_schema():
    """评测集必须能被 Suite schema 加载；版本溯源字段不能让整个文件读不进来。"""

    from learning_agent.evaluation.schemas import Suite

    suite = Suite.model_validate_json(
        (ROOT / "evals" / "learning-core-v4.json").read_text(encoding="utf-8")
    )
    assert suite.id == "learning-core-v4"
    assert suite.supersedes == "3"
    assert len(suite.cases) == 28
