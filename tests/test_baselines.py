import asyncio

from test_feedback import deliver
from test_learning import Scripted

from learning_agent.config import Settings
from learning_agent.evaluation.baselines import baseline_plan, build_results, run_baselines
from learning_agent.evaluation.runner import run_experiment
from learning_agent.evaluation.schemas import Suite
from learning_agent.provider import OpenAIProvider


class PlainModel:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def step(self, history, instructions, tools, allow_tools):
        self.calls.append(
            {
                "history": [dict(message) for message in history],
                "instructions": instructions,
                "tools": tools,
                "allow_tools": allow_tools,
            }
        )
        return {"role": "assistant", "content": self.answers.pop(0)}, {
            "total_tokens": 7,
            "prompt_tokens": 5,
            "completion_tokens": 2,
        }


def suite(actions=None):
    return Suite.model_validate(
        {
            "id": "baseline-test",
            "cases": [
                {
                    "id": "case",
                    "family": "family",
                    "turns": [
                        {
                            "id": "turn",
                            "input": {"question": "近似为什么要说明误差条件？"},
                            "gold": {"evidence_targets": [{"asset": "notes"}]},
                            "actions": actions or [],
                        }
                    ],
                }
            ],
        }
    )


def test_baselines_share_cases_and_fixed_rag_records_evidence(tmp_path):
    note = tmp_path / "notes.md"
    note.write_text("近似需要明确误差条件和适用区间。相对误差以真值为分母。")
    model = PlainModel(["直接回答", "基于资料的固定 RAG 回答"])
    result = asyncio.run(
        run_baselines(
            tmp_path / "baselines",
            suite(),
            {"notes": {"path": str(note)}},
            ("direct", "fixed_rag"),
            base=Settings(_env_file=None, data_root=tmp_path),
            provider_factory=lambda _settings: model,
        )
    )

    assert [row["status"] for row in result["rows"]] == ["completed", "completed"]
    direct, rag = result["rows"]
    assert direct["trace"]["operations"][0]["input_ref"]["evidence"] == []
    assert rag["trace"]["operations"][0]["input_ref"]["evidence"]
    assert result["summary"]["direct"]["tokens"] == 7
    assert result["summary"]["fixed_rag"]["tokens"] == 7
    assert result["summary"]["fixed_rag"]["evidence_pass"] == 1
    assert all(call["tools"] == [] and call["allow_tools"] is False for call in model.calls)
    assert "资料片段" in model.calls[1]["instructions"]


def test_baseline_skips_state_and_memory_cases(tmp_path):
    model = PlainModel(["不会被调用"])
    result = asyncio.run(
        run_baselines(
            tmp_path / "baselines",
            suite(actions=[{"kind": "learning", "values": {"goal": "测试目标"}}]),
            {},
            ("direct", "fixed_rag"),
            base=Settings(_env_file=None, data_root=tmp_path),
            provider_factory=lambda _settings: model,
        )
    )

    assert [row["status"] for row in result["rows"]] == ["not_applicable", "not_applicable"]
    assert not model.calls
    assert result["summary"]["direct"]["applicable"] == 0


def test_baseline_plan_rejects_unknown_mode():
    try:
        baseline_plan(suite(), ("direct", "unknown"))
    except ValueError as error:
        assert "不支持的基线" in str(error)
    else:
        raise AssertionError("unknown baseline should fail")


def test_baseline_turn_limit_is_per_variant(tmp_path):
    note = tmp_path / "notes.md"
    note.write_text("近似需要明确误差条件。")
    model = PlainModel(["直答", "固定 RAG"])
    result = asyncio.run(
        run_baselines(
            tmp_path / "baselines",
            suite(),
            {"notes": {"path": str(note)}},
            ("direct", "fixed_rag"),
            base=Settings(_env_file=None, data_root=tmp_path),
            max_turns=1,
            provider_factory=lambda _settings: model,
        )
    )
    assert [row["status"] for row in result["rows"]] == ["completed", "completed"]
    assert len(model.calls) == 2


def test_provider_omits_empty_tools():
    settings = Settings(_env_file=None)
    request = OpenAIProvider(settings).build_request([], "回答", [], allow_tools=False)
    assert "tools" not in request.payload
    assert "tool_choice" not in request.payload


def test_results_merges_agent_and_baselines_on_shared_turns(tmp_path):
    suite_data = Suite.model_validate(
        {
            "id": "shared",
            "cases": [
                {
                    "id": "case",
                    "family": "family",
                    "turns": [{"id": "turn", "input": {"question": "解释近似"}}],
                }
            ],
        }
    )
    agent = asyncio.run(
        run_experiment(
            tmp_path / "agent",
            suite_data,
            {},
            {"base": {}},
            base=Settings(_env_file=None, data_root=tmp_path, max_steps=3),
            provider_factory=lambda _settings: Scripted([deliver("Agent 回答")]),
        )
    )
    assert agent["counts"] == {"completed": 1}

    model = PlainModel(["直答", "固定 RAG"])
    asyncio.run(
        run_baselines(
            tmp_path / "baselines",
            suite_data,
            {},
            ("direct", "fixed_rag"),
            base=Settings(_env_file=None, data_root=tmp_path),
            provider_factory=lambda _settings: model,
        )
    )
    output = tmp_path / "results.md"
    result = build_results(tmp_path / "agent", tmp_path / "baselines", output)
    assert result["shared_turns"] == 1
    assert {"direct", "fixed_rag", "learning-agent"} <= set(result["summary"])
    assert "评测结果" in output.read_text()

    variants = build_results(tmp_path / "agent", None, tmp_path / "variants.md")
    assert variants["baseline"] is None
    assert "learning-agent" in variants["summary"]

