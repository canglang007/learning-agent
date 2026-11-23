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


def test_retrieval_v4_targets_stay_inside_lecture_time_ranges():
    """整讲语料没有切片偏移，时间轴就是原片时间轴。

    两条不变量：区间必须在资产时长内（v2 的 859 秒那种永假锚点不能再出现），
    且不能标 ``time_basis=original``——评测脚本里的资产表没有 ``source_start_seconds``，
    标了 original 会走到"拿不到偏移就判不匹配"的分支，静默变成永远漏检。
    """

    suite = json.loads((ROOT / "evals" / "retrieval-v4.json").read_text(encoding="utf-8"))
    durations = {"cs_video": 5223.0, "zh_video": 6120.0}
    identifiers, video_targets = set(), 0
    for query in suite["queries"]:
        assert query["id"] not in identifiers, f"重复的题目 id：{query['id']}"
        identifiers.add(query["id"])
        assert query.get("targets_mode", "any") in {"any", "all"}
        for target in query.get("expected", []):
            seconds = target.get("seconds")
            if not seconds:
                continue
            video_targets += 1
            assert target.get("time_basis", "clip") == "clip", query["id"]
            low, high = seconds
            assert 0 <= low < high <= durations[target["asset"]], (query["id"], seconds)
    assert video_targets >= 20, "视频目标太少，扩样本的意义就没了"


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


def test_retrieval_v5_covers_every_new_lecture_and_keeps_v4_intact():
    """v5 是在 v4 上累加，不是换掉：53 题必须原样在、每讲都得有题、跨讲题要有双证据。

    语料从 2 讲扩到 7 讲之后，最容易犯的错是"加了材料但题还压在两讲上"——
    那样新下的 4.2 小时只是占磁盘，评测根本没看见它。
    """

    v4 = json.loads((ROOT / "evals" / "retrieval-v4.json").read_text(encoding="utf-8"))
    v5 = json.loads((ROOT / "evals" / "retrieval-v5.json").read_text(encoding="utf-8"))
    by_id = {query["id"]: query for query in v5["queries"]}
    assert len(by_id) == len(v5["queries"]), "v5 里有重复 id"
    assert v5["supersedes"] == 4 and v5["version"] == 5

    # v4 的每一题都在 v5 里，且内容完全一致——扩样本不能顺手改旧题的标注。
    for query in v4["queries"]:
        assert query["id"] in by_id, f"v5 丢了 v4 的题：{query['id']}"
        assert by_id[query["id"]] == query, f"v5 改动了 v4 的题：{query['id']}"

    durations = {
        "cs_video": 5223.0,
        "cs4_video": 5180.4,
        "cs9_video": 4676.6,
        "cs10_video": 5129.6,
        "zh_video": 6120.0,
        "hy_flash": 2978.7,
        "hy_kv": 2313.5,
    }
    per_asset = {}
    for query in v5["queries"]:
        assert query.get("targets_mode", "any") in {"any", "all"}
        for target in query.get("expected", []):
            seconds = target.get("seconds")
            if not seconds:
                continue
            per_asset[target["asset"]] = per_asset.get(target["asset"], 0) + 1
            assert target.get("time_basis", "clip") == "clip", query["id"]
            low, high = seconds
            limit = durations[target["asset"]]
            assert 0 <= low < high <= limit, (query["id"], target["asset"], seconds)

    # 新增的 5 讲每讲至少 5 题，否则"覆盖新素材"名不副实。
    for asset in ("cs4_video", "cs9_video", "cs10_video", "hy_flash", "hy_kv"):
        assert per_asset.get(asset, 0) >= 5, f"{asset} 只有 {per_asset.get(asset, 0)} 道题"

    # 跨讲对照题必须是双证据 + all，任一证据缺席就不算命中。
    cross = [q for q in v5["queries"] if q["id"].startswith("cross_")]
    assert len(cross) >= 4, "跨讲对照题少于 4 道"
    for query in cross:
        assert query.get("targets_mode") == "all", query["id"]
        assert len(query["expected"]) >= 2, query["id"]

    # 负例要够用：v4 只有 3 道（5.7%），v5 至少 8 道。
    # 判定口径按"没有可标注目标"算，而不是看 negative 字段——v4 的 absent_office
    # 就是用 annotation_evidence 说明不可达、没标 negative，检查器也兼容这种写法。
    negatives = [q for q in v5["queries"] if not q.get("expected")]
    assert len(negatives) >= 8
    # 只要求 v5 新增的负例写清依据；v4 的 absent_gpu/absent_exam 是历史题，
    # 扩样本时不回头改旧数据。
    v4_ids = {query["id"] for query in v4["queries"]}
    for query in negatives:
        assert query["expected"] == [], query["id"]
        if query["id"] not in v4_ids:
            assert query.get("annotation_evidence"), f"{query['id']} 没写为什么判为不存在"
