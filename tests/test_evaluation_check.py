"""评测完整性检查脚本本身的回归。

这道闸破一次，评测集就能被拿去迎合结论——所以它自己要有测试。
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "check_evaluation", ROOT / "scripts/check_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = load()


def write(root: Path, name: str, payload: dict) -> None:
    (root / "evals").mkdir(parents=True, exist_ok=True)
    (root / "evals" / name).write_text(json.dumps(payload, ensure_ascii=False))


def suite() -> dict:
    return {
        "id": "demo-suite",
        "version": 1,
        "rubric_version": 1,
        "cases": [
            {
                "id": "c1",
                "split": "dev",
                "turns": [{"id": "t1", "input": {"question": "q"}, "gold": {"answer": "a"}}],
            }
        ],
    }


def baseline() -> dict:
    return {
        "schema_version": 1,
        "source": {"experiment": "demo"},
        "scope": {"turns": 1},
        "metrics": dict.fromkeys(CHECK.REQUIRED_METRICS, 0),
        "thresholds": {"min_review_coverage": 0.5},
    }


def retrieval(version: int, queries: list[dict], supersedes: int | None = None) -> dict:
    payload = {"id": "retrieval", "version": version, "queries": queries}
    if supersedes is not None:
        payload["supersedes"] = supersedes
    return payload


def base_queries() -> list[dict]:
    return [
        {
            "id": "hit",
            "modality": "pdf",
            "question": "问题",
            "expected": [{"asset": "a", "pages": [1]}],
        },
        {
            "id": "neg",
            "modality": "none",
            "question": "课程里有没有",
            "expected": [],
            "negative": True,
        },
        {
            "id": "scan",
            "modality": "image",
            "question": "图里矩阵形状",
            "expected": [{"asset": "w", "pages": [1]}],
            "unretrievable": {
                "reason": "纯图页，文字层为空，chunks=0",
                "readable_by": "inspect_page",
            },
        },
    ]


def test_clean_set_passes(tmp_path):
    write(tmp_path, "suite.json", suite())
    write(tmp_path, "baseline.json", baseline())
    write(tmp_path, "retrieval.json", retrieval(1, base_queries()))
    assert CHECK.main(tmp_path) == 0


def test_reason_that_serves_a_conclusion_is_rejected(tmp_path, capsys):
    queries = base_queries()
    queries[2]["unretrievable"]["reason"] = "为了让两份材料都不可文字检索"
    write(tmp_path, "suite.json", suite())
    write(tmp_path, "baseline.json", baseline())
    write(tmp_path, "retrieval.json", retrieval(1, queries))
    assert CHECK.main(tmp_path) == 1
    output = capsys.readouterr().out
    assert "意图性表述" in output
    assert "evaluation-integrity" in output


def test_unscored_query_needs_an_owner(tmp_path, capsys):
    queries = base_queries()
    del queries[2]["unretrievable"]["readable_by"]
    write(tmp_path, "suite.json", suite())
    write(tmp_path, "baseline.json", baseline())
    write(tmp_path, "retrieval.json", retrieval(1, queries))
    assert CHECK.main(tmp_path) == 1
    assert "readable_by" in capsys.readouterr().out


def test_empty_target_needs_negative_marker(tmp_path, capsys):
    queries = base_queries()
    queries[1].pop("negative")
    write(tmp_path, "suite.json", suite())
    write(tmp_path, "baseline.json", baseline())
    write(tmp_path, "retrieval.json", retrieval(1, queries))
    assert CHECK.main(tmp_path) == 1
    assert "expected 为空" in capsys.readouterr().out

    # 已写明标注依据的负例同样可以，不必为了过检查去改数据。
    queries[1]["annotation_evidence"] = "讲义与讲稿均无此信息"
    write(tmp_path, "retrieval.json", retrieval(1, queries))
    assert CHECK.main(tmp_path) == 0


def test_superseded_version_is_only_checked_for_shape(tmp_path):
    """历史版本不该被新规则回溯判错，否则每次收紧规则都得改旧数据。"""
    legacy = [
        {"id": "old", "question": "旧题"},  # 没有 modality、没有 expected
    ]
    write(tmp_path, "suite.json", suite())
    write(tmp_path, "baseline.json", baseline())
    write(tmp_path, "retrieval-v1.json", retrieval(1, legacy))
    write(tmp_path, "retrieval-v2.json", retrieval(2, base_queries(), supersedes=1))
    assert CHECK.main(tmp_path) == 0


def test_same_query_id_in_different_versions_is_not_a_duplicate(tmp_path):
    """id 只在单个版本内唯一；跨版本重复是版本演进，不是错误。"""
    write(tmp_path, "suite.json", suite())
    write(tmp_path, "baseline.json", baseline())
    write(tmp_path, "retrieval-v1.json", retrieval(1, base_queries()))
    write(tmp_path, "retrieval-v2.json", retrieval(2, base_queries(), supersedes=1))
    assert CHECK.main(tmp_path) == 0
