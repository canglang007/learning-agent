"""The quality gate must fail closed: no review, no comparison, no silent pass."""

import json

import pytest

from learning_agent.evaluation.gate import evaluate, make_baseline, summarize


def report_payload():
    def row(case, turn, *, variant="baseline", evidence="pass"):
        return {
            "variant": variant,
            "case_id": case,
            "turn_id": turn,
            "status": "completed",
            "runtime_state": "completed",
            "checks": [
                {"name": "runtime_delivery", "status": "pass"},
                {"name": "evidence_0", "status": evidence},
            ],
            "review": {
                "status": "complete",
                "criteria": {"accuracy": "pass", "clarity": "pass"},
                "serious_error": 0,
            },
        }

    return {
        "schema_version": 1,
        "manifest": {
            "suite_version": "1",
            "rubric_version": "1",
            "trials": [{"case": {"turns": [{"id": "explain"}, {"id": "followup"}]}}],
        },
        "usage": {"main_model": {"attempts": 2, "tokens": {"total_tokens": 100}}},
        "median_wall_s": 10.0,
        "rows": [row("case-a", "explain"), row("case-a", "followup")],
    }


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "report.json").write_text(json.dumps(report_payload(), ensure_ascii=False))
    return root


@pytest.fixture
def baseline(experiment):
    return make_baseline(experiment, source="unit-test")


def test_same_report_passes(baseline, experiment):
    result = evaluate(baseline, summarize(experiment))
    assert result["result"] == "passed"
    assert result["metric_deltas"]["content_pass"] == 0


def test_delivery_regression_is_caught(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["rows"][1]["status"] = "not_started"
    payload["rows"][1]["runtime_state"] = None
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "regressed"
    assert {item["metric"] for item in result["regressions"]} >= {"delivered"}


def test_evidence_and_content_regressions_are_caught(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["rows"][0]["checks"][1]["status"] = "fail"
    payload["rows"][1]["review"]["criteria"]["accuracy"] = "fail"
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "regressed"
    assert {item["metric"] for item in result["regressions"]} >= {
        "evidence_pass",
        "content_pass",
    }


def test_serious_error_is_never_offset_by_other_gains(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["rows"][0]["review"]["serious_error"] = 1
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "regressed"
    assert any(item["metric"] == "serious_errors" for item in result["regressions"])


def test_missing_review_is_insufficient_evidence(baseline, experiment, tmp_path):
    payload = report_payload()
    for row in payload["rows"]:
        row["review"] = None
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "insufficient_evidence"
    assert "未复核" in result["reason"]


def test_suite_change_is_incomparable(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["manifest"]["suite_version"] = "2"
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "incomparable"
    assert "suite_version" in result["scope_deltas"]


def test_case_set_change_is_incomparable(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["rows"][1]["case_id"] = "case-b"
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "incomparable"
    assert "cases" in result["scope_deltas"]


def test_cost_and_latency_are_not_quality_signals(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["usage"]["main_model"]["tokens"]["total_tokens"] = 999999
    payload["median_wall_s"] = 9999.0
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "passed"
    assert result["cost_context"]["main_model_tokens"] == 999999


def test_improvement_is_reported_without_failing(baseline, experiment, tmp_path):
    payload = report_payload()
    payload["rows"][0]["checks"][1]["status"] = "pass"
    payload["rows"][1]["review"]["serious_error"] = 0
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(payload))
    result = evaluate(baseline, summarize(other))
    assert result["result"] == "passed"
    assert isinstance(result["improvements"], list)


def test_snapshot_refuses_unreviewed_report(experiment, tmp_path):
    payload = report_payload()
    for row in payload["rows"]:
        row["review"] = None
    root = tmp_path / "unreviewed"
    root.mkdir()
    (root / "report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="内容复核"):
        make_baseline(root)


def test_variant_scope_is_part_of_comparison(experiment, tmp_path):
    mixed = report_payload()
    mixed["rows"][1]["variant"] = "rrf"
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "report.json").write_text(json.dumps(mixed))
    only_baseline = summarize(root, variant="baseline")
    assert only_baseline["scope"]["variant"] == "baseline"
    assert len(only_baseline["scope"]["turns"]) == 1
