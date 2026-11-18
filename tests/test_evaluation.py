import asyncio
import json

import pytest
from test_feedback import deliver
from test_learning import Scripted

from learning_agent.config import Settings
from learning_agent.evaluation.ledger import EvaluationBudgetExceeded, Ledger
from learning_agent.evaluation.report import compare, report
from learning_agent.evaluation.review import save_review
from learning_agent.evaluation.runner import plan, run_experiment
from learning_agent.evaluation.schemas import Suite
from learning_agent.provider import ModelError


def suite():
    return Suite.model_validate(
        {
            "id": "small",
            "cases": [
                {
                    "id": "explain",
                    "family": "approx",
                    "turns": [
                        {
                            "id": "one",
                            "input": {"question": "解释近似"},
                            "gold": {"required_concepts": ["PRIVATE_GOLD_SENTINEL"]},
                        },
                        {
                            "id": "two",
                            "input": {
                                "question": "给个例子",
                                "feedback": {"kind": "example", "target_turn": "one"},
                            },
                        },
                    ],
                }
            ],
        }
    )


def test_runtime_runner_input_manifest_no_gold_leak_and_continue_no_resend(tmp_path):
    model = Scripted([deliver("受控近似误差"), deliver("1.001近似为1")])
    output = tmp_path / "experiment"
    settings = Settings(_env_file=None, data_root=tmp_path, max_steps=3)
    result = asyncio.run(
        run_experiment(
            output, suite(), {}, {"base": {}}, base=settings, provider_factory=lambda _: model
        )
    )
    assert result["counts"] == {"completed": 2}
    assert result["usage"]["main_model"]["attempts"] == 2
    assert result["usage"]["main_model"]["tokens"]["total_tokens"] == 20
    assert "PRIVATE_GOLD_SENTINEL" not in json.dumps(model.inputs)
    trace = report(output, include_content=True)["rows"][0]["trace"]
    assert trace["operations"][0]["input_ref"]["schema_version"] == 1
    assert trace["answer"] and trace["complete"]
    again = asyncio.run(
        run_experiment(
            output,
            suite(),
            {},
            {"base": {}},
            base=settings,
            continue_run=True,
            provider_factory=lambda _: model,
        )
    )
    assert again["counts"] == {"completed": 2} and len(model.inputs) == 2


def test_budget_is_persistent_idempotent_and_blocks_before_sending(tmp_path):
    ledger = Ledger(tmp_path / "budget.db")
    ledger.put("limits", {"main_model": 1})
    ledger.reserve("op-1", "main_model")
    Ledger(ledger.path).reserve("op-1", "main_model")
    with pytest.raises(EvaluationBudgetExceeded):
        ledger.reserve("op-2", "main_model")
    model = Scripted([deliver("不会调用")])
    result = asyncio.run(
        run_experiment(
            tmp_path / "run",
            suite(),
            {},
            {"base": {}},
            base=Settings(_env_file=None),
            limits={"main_model": 0},
            provider_factory=lambda _: model,
        )
    )
    assert not model.inputs
    assert result["counts"] == {"stopped_budget": 1, "blocked": 1}


def test_unknown_result_blocks_followup_and_continue_does_not_resend(tmp_path):
    model = Scripted([ModelError("unknown", outcome="unknown")])
    output = tmp_path / "run"
    settings = Settings(_env_file=None)
    first = asyncio.run(
        run_experiment(
            output, suite(), {}, {"base": {}}, base=settings, provider_factory=lambda _: model
        )
    )
    assert first["counts"] == {"interrupted": 1, "blocked": 1}
    second = asyncio.run(
        run_experiment(
            output,
            suite(),
            {},
            {"base": {}},
            base=settings,
            provider_factory=lambda _: model,
            continue_run=True,
        )
    )
    assert second["counts"] == first["counts"] and len(model.inputs) == 1


def test_review_revisions_regrade_without_model_and_paired_comparison(tmp_path):
    model = Scripted([deliver("a"), deliver("b"), deliver("c"), deliver("d")])
    asyncio.run(
        run_experiment(
            tmp_path / "run",
            suite(),
            {},
            {"before": {}, "after": {"retrieval_mode": "bm25"}},
            base=Settings(_env_file=None),
            provider_factory=lambda _: model,
        )
    )
    for variant, status in (("before", "fail"), ("after", "pass")):
        save_review(
            tmp_path / "run",
            {
                "trial_id": variant + "-explain-0",
                "turn_id": "one",
                "reviewer_kind": "developer_agent",
                "reviewer": "test",
                "criteria": {"concept_0": status},
                "scores": {"accuracy": 0 if status == "fail" else 3},
                "serious_error": status == "fail",
                "reasons": ["有依据的复核"],
                "evidence_refs": ["answer"],
            },
        )
    result = compare(tmp_path / "run", variants=["before", "after"])
    assert result["pairs"][0]["result"] == "improved" and len(model.inputs) == 4
    assert report(tmp_path / "run")["reviewed"] == 2


def test_family_split_and_variant_paths_are_validated():
    spec = suite().model_dump()
    spec["cases"].append({**spec["cases"][0], "id": "held", "split": "holdout"})
    with pytest.raises(ValueError):
        Suite.model_validate(spec)
    with pytest.raises(ValueError):
        plan(suite(), {"../escape": {}})


def test_video_target_requires_complete_anchors_and_normalizes_source_time():
    from learning_agent.evaluation.graders import matches

    assets = {"video": {"id": "v", "source_start_seconds": 100}}
    target = {"asset": "video", "anchors": [135, 152], "time_basis": "original"}
    assert (
        matches({"material_id": "v", "start_seconds": 35, "end_seconds": 36}, target, assets)
        is False
    )
    assert (
        matches({"material_id": "v", "start_seconds": 30, "end_seconds": 60}, target, assets)
        is True
    )


def test_observed_image_is_not_same_as_image_in_actual_input():
    from learning_agent.evaluation.graders import grade

    trace = {
        "state": "completed",
        "answer": "解释",
        "messages": [{"role": "assistant", "kind": "answer", "content": "解释"}],
        "tools": [{"name": "inspect_page", "observation": {"material_id": "pdf", "page": 10}}],
        "operations": [
            {
                "id": "op",
                "purpose": "main_model",
                "input_ref": {"schema_version": 1, "images": [], "evidence": []},
            }
        ],
    }
    gold = {"evidence_targets": [{"asset": "pdf", "pages": [10], "modality": "image"}]}
    missing = grade(trace, gold, {"pdf": {"id": "pdf"}})
    assert missing["checks"][1]["status"] == "fail"
    assert missing["findings"][0]["category"] == "context_selection"
    trace["operations"][0]["input_ref"]["images"] = [
        {"asset_path": "materials/pdf/page-10-full.png"}
    ]
    assert grade(trace, gold, {"pdf": {"id": "pdf"}})["checks"][1]["status"] == "pass"


def test_invalid_judge_is_recorded_and_not_automatically_resent(tmp_path):
    import httpx

    from learning_agent.evaluation.review import judge
    from learning_agent.provider import OpenAIProvider

    spec = suite().model_dump()
    spec["cases"][0]["turns"] = spec["cases"][0]["turns"][:1]
    model = Scripted([deliver("近似需要误差条件")])
    root = tmp_path / "run"
    asyncio.run(
        run_experiment(
            root,
            Suite.model_validate(spec),
            {},
            {"base": {}},
            base=Settings(_env_file=None),
            provider_factory=lambda _: model,
        )
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "not structured"}}]})

    settings = Settings(_env_file=None)
    provider = OpenAIProvider(settings, httpx.MockTransport(handler))
    asyncio.run(judge(root, settings, provider=provider))
    asyncio.run(judge(root, settings, provider=provider))
    assert len(requests) == 1 and report(root)["reviewed"] == 0
    assert Ledger(root / "experiment.db").records("judge:")[0]["state"] == "grader_error"


def test_readonly_review_api_and_path_boundaries(tmp_path):
    from fastapi.testclient import TestClient

    from learning_agent.app import create_app

    model = Scripted([deliver("近似")] * 2)
    root = tmp_path / "evaluations" / "test-run"
    asyncio.run(
        run_experiment(
            root,
            suite(),
            {},
            {"base": {}},
            base=Settings(_env_file=None),
            provider_factory=lambda _: model,
        )
    )
    before = (root / "experiment.db").stat().st_mtime_ns
    app = create_app(
        Settings(
            _env_file=None, data_root=tmp_path / "product", evaluation_root=tmp_path / "evaluations"
        ),
        Scripted([]),
    )
    with TestClient(app) as client:
        assert client.get("/evaluation").status_code == 200
        assert client.get("/api/evaluations").json()[0]["id"] == "test-run"
        view = client.get("/api/evaluations/test-run").json()
        assert view["reviewed"] == 0 and "answer" not in view["rows"][0]
        detail = client.get("/api/evaluations/test-run/trials/base-explain-0").json()
        assert detail["rows"][0]["answer"] == "近似"
        assert client.get("/api/evaluations/not-found").status_code == 404
        assert client.get("/api/evaluations/test-run/images/../../.env").status_code in {404, 422}
    assert (root / "experiment.db").stat().st_mtime_ns == before


def test_database_thread_reads_and_nested_transactions_finish(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from learning_agent.store import Store

    store = Store(tmp_path / "database")
    sid = store.space("并发测试")["id"]

    def work(i):
        for _ in range(20):
            assert store.materials(sid) == []
            with store.connect() as db:
                assert db.execute("SELECT count(*) FROM spaces").fetchone()[0] == len(
                    store.spaces()
                )
        return i

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work, i) for i in range(4)]
        assert [f.result(timeout=5) for f in futures] == list(range(4))


def test_core_suite_and_hidden_key_configuration(tmp_path, capsys):
    from pathlib import Path

    from learning_agent.launch import doctor, write_config

    core = Suite.model_validate_json(Path("evals/learning-core-v1.json").read_text())
    assert len(core.cases) == 28 and sum(len(c.turns) for c in core.cases) == 56
    assert sum(c.split == "holdout" for c in core.cases) == 10
    config = tmp_path / ".env"
    config.write_text("LA_MAX_STEPS=4\n")
    write_config(config, {"LA_API_KEY": "test-key-never-print", "LA_MODEL": "example-model"})
    settings = Settings(_env_file=config)
    assert settings.max_steps == 4 and settings.api_key.get_secret_value() == "test-key-never-print"
    print(doctor(settings))
    assert "test-key-never-print" not in capsys.readouterr().out


def test_memory_and_user_state_initialization_use_real_product_interfaces(tmp_path):
    from pathlib import Path

    from learning_agent.evaluation.schemas import Suite

    core = Suite.model_validate_json(Path("evals/learning-core-v1.json").read_text())
    wanted = {"memory_attach", "memory_disable", "memory_forget", "state_correction"}
    selected = core.model_copy(update={"cases": [c for c in core.cases if c.id in wanted]})
    model = Scripted([deliver("围绕当前问题解释") for _ in range(8)])
    result = asyncio.run(
        run_experiment(
            tmp_path / "memory-eval",
            selected,
            {},
            {"base": {}},
            split="all",
            base=Settings(_env_file=None),
            provider_factory=lambda _: model,
        )
    )
    assert result["counts"] == {"completed": 8}
    checks = [
        c
        for r in result["rows"]
        for c in r["checks"]
        if c["name"].startswith("state:") or c["name"] == "memory_selected"
    ]
    assert len(checks) == 5 and all(c["status"] == "pass" for c in checks)


def test_component_metrics_preserve_annotation_and_actual_mode(tmp_path):
    from learning_agent.evaluation.retrieval import evaluate_retrieval
    from learning_agent.evaluation.runner import prepare

    root = tmp_path / "component"
    settings = Settings(_env_file=None, data_root=root / "corpus", retrieval_mode="bm25")
    path = tmp_path / "notes.txt"
    path.write_text("matrix multiplication matrix product dot product")
    corpus = prepare(root / "corpus", {"notes": {"path": str(path)}}, settings)
    ledger = Ledger(root / "experiment.db")
    ledger.put("corpus", corpus)
    ledger.put("limits", {})
    result = evaluate_retrieval(
        root,
        [
            {"id": "matrix", "question": "matrix multiplication", "expected": [{"asset": "notes"}]},
            {"id": "missing", "question": "unknown-never-seen-word", "expected": []},
        ],
        settings,
    )
    assert result["results"][0]["annotated_hit_at_5"] == 1
    assert result["results"][0]["annotated_target_queries"] == 1
    assert result["results"][1]["actual_modes"] == {"bm25": 2}
    assert result["results"][1]["rows"][0]["dense_status"] != "ready"


def test_unretrievable_target_is_listed_without_counting_as_a_miss(tmp_path):
    """纯图页没有文字层，检索永远够不着；它该单列，不该拉低召回。"""
    from learning_agent.evaluation.retrieval import evaluate_retrieval
    from learning_agent.evaluation.runner import prepare

    root = tmp_path / "component"
    settings = Settings(_env_file=None, data_root=root / "corpus", retrieval_mode="bm25")
    path = tmp_path / "notes.txt"
    path.write_text("matrix multiplication matrix product dot product")
    corpus = prepare(root / "corpus", {"notes": {"path": str(path)}}, settings)
    ledger = Ledger(root / "experiment.db")
    ledger.put("corpus", corpus)
    ledger.put("limits", {})
    result = evaluate_retrieval(
        root,
        [
            {"id": "matrix", "question": "matrix multiplication", "expected": [{"asset": "notes"}]},
            {
                "id": "scanned",
                "question": "matrix multiplication",
                "expected": [{"asset": "notes", "pages": [1]}],
                "unretrievable": {"reason": "纯图页无文字层"},
            },
            {"id": "missing", "question": "dot product", "expected": []},
        ],
        settings,
    )
    first = result["results"][0]
    assert first["annotated_target_queries"] == 1
    assert first["annotated_hit_at_5"] == 1
    assert first["unretrievable_queries"] == ["scanned"]
    assert first["negative_queries"] == 1
    assert first["negative_top1_scores"]


def test_negative_score_overlap_flags_scores_as_unusable_for_refusal(tmp_path):
    """负例最高分不低于命中最低分时，必须标出分数不能用来拒答。"""
    from learning_agent.evaluation.retrieval import evaluate_retrieval
    from learning_agent.evaluation.runner import prepare

    root = tmp_path / "component"
    settings = Settings(_env_file=None, data_root=root / "corpus", retrieval_mode="bm25")
    path = tmp_path / "notes.txt"
    path.write_text("matrix multiplication matrix product dot product")
    corpus = prepare(root / "corpus", {"notes": {"path": str(path)}}, settings)
    ledger = Ledger(root / "experiment.db")
    ledger.put("corpus", corpus)
    ledger.put("limits", {})
    result = evaluate_retrieval(
        root,
        [
            {"id": "matrix", "question": "matrix multiplication", "expected": [{"asset": "notes"}]},
            {"id": "missing", "question": "matrix multiplication", "expected": []},
        ],
        settings,
    )
    first = result["results"][0]
    assert first["negative_queries"] == 1
    # 负例和正例走同一次检索，分数相同，于是重叠必然为真。
    assert first["negative_score_overlap"] is True


def test_unlabelled_export_does_not_leak_answer_or_fake_quality(tmp_path):
    from learning_agent.evaluation.collector import export_run

    model = Scripted([deliver("PRIVATE_ANSWER")] * 2)
    root = tmp_path / "export-test"
    asyncio.run(
        run_experiment(
            root,
            suite(),
            {},
            {"base": {}},
            base=Settings(_env_file=None),
            provider_factory=lambda _: model,
        )
    )
    row = report(root, include_content=True)["rows"][0]
    workspace = root / "trials" / row["trial_id"] / "workspace"
    output = export_run(workspace, row["trace"]["run_id"])
    assert "PRIVATE_ANSWER" not in json.dumps(output)
    assert output["quality"] == "needs_review_no_reference_labels"
    assert (
        export_run(workspace, row["trace"]["run_id"], include_content=True)["trace"]["answer"]
        == "PRIVATE_ANSWER"
    )


def test_invalid_numeric_grader_is_not_answer_failure():
    from learning_agent.evaluation.graders import grade

    trace = {
        "state": "completed",
        "answer": "x=1",
        "messages": [{"role": "assistant", "kind": "answer", "content": "x=1"}],
    }
    result = grade(trace, {"numerical_checks": [{"pattern": "[", "value": 1}]}, {})
    numeric = next(c for c in result["checks"] if c["name"] == "numeric_0")
    assert numeric["status"] == "needs_review" and "grader_error" in numeric["reason"]


def test_concurrent_judges_cannot_duplicate_paid_request(tmp_path):
    import httpx

    from learning_agent.evaluation.review import judge
    from learning_agent.provider import OpenAIProvider

    spec = suite().model_dump()
    spec["cases"][0]["turns"] = spec["cases"][0]["turns"][:1]
    root = tmp_path / "judge-lock"
    settings = Settings(_env_file=None)
    asyncio.run(
        run_experiment(
            root,
            Suite.model_validate(spec),
            {},
            {"base": {}},
            base=settings,
            provider_factory=lambda _: Scripted([deliver("近似")]),
        )
    )

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []

        async def handler(request):
            requests.append(request)
            entered.set()
            await release.wait()
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "invalid grading response"}}]}
            )

        provider = OpenAIProvider(settings, httpx.MockTransport(handler))
        first = asyncio.create_task(judge(root, settings, provider=provider))
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ValueError, match="已有裁判"):
            await judge(root, settings, provider=provider)
        release.set()
        await first
        assert len(requests) == 1

    asyncio.run(scenario())


def test_judge_budget_extension_is_recorded(tmp_path):
    from learning_agent.evaluation.ledger import Ledger
    from learning_agent.evaluation.review import extend_judge_budget

    ledger = Ledger(tmp_path / "experiment.db")
    ledger.put("limits", {"main_model": 10, "judge": 12})
    previous = extend_judge_budget(tmp_path, 60)
    assert previous == 12
    assert ledger.get("limits")["judge"] == 60
    extensions = ledger.records("limits-extension:")
    assert len(extensions) == 1
    assert extensions[0]["purpose"] == "judge" and extensions[0]["to"] == 60
    with pytest.raises(ValueError):
        extend_judge_budget(tmp_path, -1)


def test_summary_evidence_exposes_quoted_context_to_the_judge(tmp_path):
    """judge 复核输入必须包含模型实际收到的摘要条目，否则如实引用摘要
    原话会被误判成无依据断言（v4 全量复核实测）。"""
    import sqlite3

    from learning_agent.evaluation.review import summary_evidence

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = json.dumps(
        {
            "items": [
                {
                    "kind": "user_statement",
                    "text": "用户的目标是比较近似的绝对误差。",
                    "user_quote": "我的目标是比较近似的绝对误差。",
                }
            ]
        },
        ensure_ascii=False,
    )
    with sqlite3.connect(workspace / "workspace.db") as db:
        db.execute("CREATE TABLE context_summaries (id TEXT PRIMARY KEY, payload TEXT)")
        db.execute("INSERT INTO context_summaries VALUES ('s1', ?)", (payload,))
    manifests = [{"schema_version": 1, "summary_id": "s1", "evidence": []}]
    items = summary_evidence(workspace, manifests)
    assert len(items) == 1
    assert items[0]["user_quote"] == "我的目标是比较近似的绝对误差。"
    assert items[0]["summary_id"] == "s1"
    # 无摘要 id / 无表 / 库缺失时如实返回空，不制造证据。
    assert summary_evidence(workspace, [{"schema_version": 1}]) == []
