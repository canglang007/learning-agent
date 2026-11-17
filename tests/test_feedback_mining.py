import asyncio
import json

import pytest
from pydantic import ValidationError
from test_feedback import deliver
from test_learning import Scripted

from learning_agent.config import Settings
from learning_agent.evaluation.feedback import build_suite, mine, normalize
from learning_agent.evaluation.report import report
from learning_agent.evaluation.runner import run_experiment
from learning_agent.evaluation.schemas import Suite


def usage_suite(kind="stuck"):
    return Suite.model_validate(
        {
            "id": "usage",
            "cases": [
                {
                    "id": "usage_case",
                    "family": "usage",
                    "turns": [
                        {
                            "id": "explain",
                            "input": {"question": "为什么近似相等要写误差"},
                        },
                        {
                            "id": "followup",
                            "input": {
                                "question": "这一步没懂，误差为什么要除以真值",
                                "feedback": {
                                    "kind": kind,
                                    "focus_text": "分母为什么取真值",
                                    "target_turn": "explain",
                                },
                            },
                        },
                    ],
                }
            ],
        }
    )


def run_usage(tmp_path, kind="stuck"):
    """跑一次带反馈的真实运行，返回它的workspace目录。"""
    output = tmp_path / "experiment"
    model = Scripted([deliver("近似要给出误差范围"), deliver("相对误差以真值为分母")])
    asyncio.run(
        run_experiment(
            output,
            usage_suite(kind),
            {},
            {"base": {}},
            base=Settings(_env_file=None, data_root=tmp_path, max_steps=2),
            provider_factory=lambda _: model,
        )
    )
    row = report(output, include_content=True)["rows"][0]
    return output / "trials" / row["trial_id"] / "workspace"


def test_mine_ranks_feedback_and_leaves_gold_for_humans(tmp_path):
    workspace = run_usage(tmp_path)
    result = mine(workspace, eval_dir=None)
    assert result["scanned"] == 2
    top = result["candidates"][0]
    assert top["feedback"]["kind"] == "stuck"
    assert top["score"] >= 4
    assert top["needs_human_gold"] is True
    case = top["suggested_case"]
    assert [turn["id"] for turn in case["turns"]] == ["explain", "followup"]
    assert case["turns"][1]["input"]["feedback"]["target_turn"] == "explain"
    assert case["turns"][1]["input"]["feedback"]["kind"] == "stuck"
    # gold 必须留空：挖掘只提供结构，概念和要求由人写
    assert all(turn["gold"]["required_concepts"] == [] for turn in case["turns"])


def test_understood_feedback_ranks_below_stuck(tmp_path):
    stuck = mine(run_usage(tmp_path / "stuck", "stuck"), eval_dir=None, min_score=-10)["candidates"]
    understood = mine(
        run_usage(tmp_path / "understood", "understood"), eval_dir=None, min_score=-10
    )["candidates"]
    stuck_top = max(item["score"] for item in stuck if item["feedback"])
    understood_top = max(item["score"] for item in understood if item["feedback"])
    assert stuck_top > understood_top


def test_mine_marks_existing_questions_as_duplicates(tmp_path):
    workspace = run_usage(tmp_path)
    directory = tmp_path / "evals"
    directory.mkdir()
    (directory / "suite.json").write_text(
        json.dumps(
            {
                "id": "existing",
                "cases": [
                    {
                        "id": "known",
                        "family": "known",
                        "turns": [
                            {"id": "t", "input": {"question": "为什么  近似相等，要写误差？"}}
                        ],
                    }
                ],
            }
        )
    )
    assert normalize("为什么  近似相等，要写误差？") == normalize("为什么近似相等要写误差")
    # 阈值放到最低，才能看见被去重压到负分的记录
    result = mine(workspace, eval_dir=directory, min_score=-10)
    duplicates = [item for item in result["candidates"] if item["duplicate_of"]]
    assert duplicates
    assert duplicates[0]["duplicate_of"] == "known/t"
    assert result["eval_questions"] == 1


def test_mine_never_writes_to_the_workspace(tmp_path):
    workspace = run_usage(tmp_path)
    database = workspace / "workspace.db"
    before = database.stat().st_mtime_ns
    mine(workspace, eval_dir=None)
    assert database.stat().st_mtime_ns == before


def test_build_suite_rejects_empty_candidates():
    with pytest.raises(ValidationError):
        build_suite([])


def test_locations_skip_entries_without_position():
    from learning_agent.evaluation.feedback import locations

    trace = {
        "operations": [
            {
                "input_ref": {
                    "schema_version": 1,
                    "evidence": [
                        {"tool_call_id": "x", "text": "只有文字，没有位置"},
                        {"material_id": "m1", "page": 3, "text": "有页码"},
                    ],
                    "images": [{"asset_path": "materials/m2/page-7-1.png", "page": 7}],
                }
            }
        ]
    }
    found = locations(trace, {"m1": "cs_pdf", "m2": "optimization"})
    # 图片证据排在文字前面：真的看过图比检索命中更硬
    assert [(item["material_id"], item["page"]) for item in found] == [("m2", 7), ("m1", 3)]
    # 只带 asset_path 的条目要能反解出资料ID，否则targets生成不了asset
    assert found[0]["name"] == "optimization"


def test_mine_reports_missing_workspace():
    with pytest.raises(FileNotFoundError):
        mine("/nonexistent/workspace", eval_dir=None)
