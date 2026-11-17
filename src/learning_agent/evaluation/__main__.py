import argparse
import asyncio
import json
from pathlib import Path

from learning_agent.config import Settings
from learning_agent.evaluation.baselines import baseline_plan, build_results, run_baselines
from learning_agent.evaluation.ledger import save_json
from learning_agent.evaluation.report import compare, report
from learning_agent.evaluation.review import judge, save_review
from learning_agent.evaluation.runner import configuration, plan, run_experiment
from learning_agent.evaluation.schemas import Suite
from learning_agent.provider import ModelError


def main():
    parser = argparse.ArgumentParser(description="本机统一评测；默认不发出模型请求")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--assets", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--case", action="append", dest="cases")
    run.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    run.add_argument("--variants", type=Path)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-turns", type=int, default=12)
    run.add_argument("--max-main-attempts", type=int, default=48)
    run.add_argument("--max-summary-attempts", type=int, default=4)
    run.add_argument("--max-judge-attempts", type=int, default=12)
    run.add_argument("--continue", action="store_true", dest="continue_run")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--real", action="store_true")
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--suite", type=Path, required=True)
    baseline.add_argument("--assets", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--case", action="append", dest="cases")
    baseline.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    baseline.add_argument("--variants", nargs="+", choices=["direct", "fixed_rag"], default=["direct", "fixed_rag"])
    baseline.add_argument("--max-turns", type=int, default=12)
    baseline_mode = baseline.add_mutually_exclusive_group()
    baseline_mode.add_argument("--plan", action="store_true")
    baseline_mode.add_argument("--real", action="store_true")
    results = sub.add_parser("results")
    results.add_argument("--experiment", type=Path, required=True)
    results.add_argument("--baseline", type=Path)
    results.add_argument("--output", type=Path, required=True)
    view = sub.add_parser("report")
    view.add_argument("experiment", type=Path)
    view.add_argument("--include-content", action="store_true")
    view.add_argument("--output", type=Path)
    pair = sub.add_parser("compare")
    pair.add_argument("baseline", type=Path)
    pair.add_argument("candidate", type=Path, nargs="?")
    pair.add_argument("--variants", nargs=2)
    review = sub.add_parser("regrade")
    review.add_argument("experiment", type=Path)
    review.add_argument("--reviews", type=Path)
    review.add_argument("--judge", action="store_true")
    review.add_argument("--retry-judge", action="store_true")
    review.add_argument("--limit", type=int, default=12)
    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("source", type=Path)
    legacy.add_argument("output", type=Path)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--samples", type=Path, default=Path("evals/calibration-v1.json"))
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--real", action="store_true")
    retrieval = sub.add_parser("retrieval")
    retrieval.add_argument("experiment", type=Path)
    retrieval.add_argument("--queries", type=Path, default=Path("evals/retrieval-v1.json"))
    retrieval.add_argument("--real", action="store_true")
    export = sub.add_parser("export-run")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-content", action="store_true")
    mine = sub.add_parser(
        "mine-feedback",
        help="从真实使用记录里挑回归候选：只读、不调用模型、不自动写gold",
    )
    mine.add_argument("--workspace", type=Path, required=True)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--eval-dir", type=Path, default=Path("evals"))
    mine.add_argument("--since", help="只统计该日期及之后的运行，格式YYYY-MM-DD")
    mine.add_argument("--limit", type=int, default=40)
    mine.add_argument("--min-score", type=int, default=1)
    mine.add_argument("--suite", type=Path, help="额外导出可直接交给run子命令的Suite片段")
    mine.add_argument("--queue", type=Path, help="把候选并进本机审核队列，已有状态不覆盖")
    review = sub.add_parser(
        "review-candidate", help="给候选定状态并补gold；队列是本机文件，不进仓库"
    )
    review.add_argument("--queue", type=Path, required=True)
    review.add_argument("--case-id", required=True)
    decide = review.add_mutually_exclusive_group(required=True)
    decide.add_argument("--accept", action="store_true")
    decide.add_argument("--reject", action="store_true")
    review.add_argument("--note", default="")
    review.add_argument("--concept", action="append", dest="concepts", default=[])
    review.add_argument("--forbid", action="append", dest="forbids", default=[])
    promote = sub.add_parser(
        "promote-cases", help="把已接受的候选并进suite，升版本写新文件"
    )
    promote.add_argument("--queue", type=Path, required=True)
    promote.add_argument("--suite", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--version", help="新suite版本号，不填则沿用原版本")
    quality = sub.add_parser(
        "gate", help="质量门禁：只读已冻结报告，与基线对比；不调用模型"
    )
    mode = quality.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", type=Path, help="与已有基线对比")
    mode.add_argument("--snapshot", type=Path, help="把该报告固化为新基线")
    quality.add_argument("--experiment", type=Path, required=True)
    quality.add_argument("--variant")
    quality.add_argument("--output", type=Path)
    quality.add_argument("--note", default="")
    args = parser.parse_args()
    try:
        if args.command == "run":
            suite = Suite.model_validate_json(args.suite.read_text())
            variants = json.loads(args.variants.read_text()) if args.variants else {"baseline": {}}
            bindings = json.loads(args.assets.read_text())
            trials = plan(suite, variants, args.cases, args.split, args.repeats)
            for variant in variants.values():
                configuration(Settings(), variant)
            if not args.real:
                result = {
                    "mode": "plan",
                    "trials": [t["id"] for t in trials],
                    "planned_turns": sum(len(t["case"]["turns"]) for t in trials),
                    "max_turns": args.max_turns,
                    "limits": {
                        "main_model": args.max_main_attempts,
                        "summary": args.max_summary_attempts,
                        "judge": args.max_judge_attempts,
                    },
                    "assets": sorted(bindings),
                    "network_calls": 0,
                }
            else:
                if not Settings().api_key.get_secret_value():
                    parser.error("请先在本机配置LA_API_KEY")
                result = asyncio.run(
                    run_experiment(
                        args.output,
                        suite,
                        bindings,
                        variants,
                        cases=args.cases,
                        split=args.split,
                        repeats=args.repeats,
                        max_turns=args.max_turns,
                        limits={
                            "main_model": args.max_main_attempts,
                            "summary": args.max_summary_attempts,
                            "judge": args.max_judge_attempts,
                        },
                        continue_run=args.continue_run,
                    )
                )
        elif args.command == "baseline":
            suite = Suite.model_validate_json(args.suite.read_text())
            bindings = json.loads(args.assets.read_text())
            baseline_trials = baseline_plan(suite, args.variants, args.cases, args.split)
            if not args.real:
                result = {
                    "mode": "plan",
                    "variants": args.variants,
                    "cases": [item["case"].id for item in baseline_trials],
                    "planned_turns": sum(len(item["case"].turns) for item in baseline_trials),
                    "max_turns": args.max_turns,
                    "network_calls": 0,
                }
            else:
                if not Settings().api_key.get_secret_value():
                    parser.error("请先在本机配置LA_API_KEY")
                full = asyncio.run(
                    run_baselines(
                        args.output,
                        suite,
                        bindings,
                        args.variants,
                        cases=args.cases,
                        split=args.split,
                        max_turns=args.max_turns,
                    )
                )
                result = {
                    "summary": full["summary"],
                    "recorded": len(full["rows"]),
                    "output": str(args.output),
                }
        elif args.command == "results":
            result = build_results(args.experiment, args.baseline, args.output)
        elif args.command == "report":
            result = report(args.experiment, include_content=args.include_content)
            if args.output:
                save_json(args.output, result)
        elif args.command == "compare":
            result = compare(args.baseline, args.candidate, variants=args.variants)
        elif args.command == "regrade":
            if args.reviews:
                for value in json.loads(args.reviews.read_text()):
                    save_review(args.experiment, value)
            result = (
                asyncio.run(
                    judge(args.experiment, Settings(), limit=args.limit, retry=args.retry_judge)
                )
                if args.judge
                else report(args.experiment)
            )
        elif args.command == "export-run":
            from learning_agent.evaluation.collector import export_run

            result = export_run(args.workspace, args.run_id, include_content=args.include_content)
            save_json(args.output, result)
        elif args.command == "mine-feedback":
            from learning_agent.evaluation import queue as queue_store
            from learning_agent.evaluation.feedback import build_suite, mine

            result = mine(
                args.workspace,
                eval_dir=args.eval_dir,
                since=args.since,
                limit=args.limit,
                min_score=args.min_score,
            )
            save_json(args.output, result)
            if args.suite:
                if not result["candidates"]:
                    parser.error("没有达到阈值的候选，不导出空Suite")
                save_json(args.suite, build_suite(result["candidates"]))
            merged = None
            if args.queue:
                existing = queue_store.load(args.queue)
                merged = queue_store.merge(existing, result["candidates"])
                queue_store.save(args.queue, existing)
            print(
                json.dumps(
                    {
                        "scanned": result["scanned"],
                        "skipped": result["skipped"],
                        "candidates": len(result["candidates"]),
                        "output": str(args.output),
                        "suite": str(args.suite) if args.suite else None,
                        "queue": merged,
                        "network_calls": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(0)
        elif args.command == "review-candidate":
            from learning_agent.evaluation import queue as queue_store

            data = queue_store.load(args.queue)
            entry = queue_store.review(
                data,
                args.case_id,
                "accepted" if args.accept else "rejected",
                note=args.note,
                concepts=args.concepts,
                forbidden=args.forbids,
            )
            queue_store.save(args.queue, data)
            print(
                json.dumps(
                    {
                        "case_id": entry["case_id"],
                        "status": entry["status"],
                        "reviewed_at": entry["reviewed_at"],
                        "gold_ready": queue_store.gold_ready(entry.get("case") or {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(0)
        elif args.command == "promote-cases":
            from learning_agent.evaluation import queue as queue_store

            data = queue_store.load(args.queue)
            payload, added, blocked = queue_store.promote(
                data,
                json.loads(args.suite.read_text()),
                version=args.version,
            )
            if not added:
                parser.error("没有可promote的候选：要么没接受，要么gold还是空的")
            save_json(args.output, payload)
            print(
                json.dumps(
                    {
                        "suite": payload["id"],
                        "version": payload["version"],
                        "cases": len(payload["cases"]),
                        "added": added,
                        "blocked_empty_gold": blocked,
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(0)
        elif args.command == "retrieval":
            if not args.real:
                parser.error("检索组件可能调用向量服务，需要显式--real；不会调用讲解模型")
            from learning_agent.evaluation.retrieval import evaluate_retrieval

            result = evaluate_retrieval(
                args.experiment, json.loads(args.queries.read_text())["queries"], Settings()
            )
        elif args.command == "gate":
            from learning_agent.evaluation.gate import (
                EXIT_CODES,
                gate,
                make_baseline,
                write_baseline,
            )

            if args.snapshot:
                result = make_baseline(
                    args.experiment, variant=args.variant, note=args.note
                )
                write_baseline(args.snapshot, result)
                exit_code = 0
            else:
                result = gate(args.baseline, args.experiment, variant=args.variant)
                exit_code = EXIT_CODES.get(result["result"], 1)
            if args.output:
                save_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(exit_code)
        elif args.command == "calibrate":
            if not args.real:
                parser.error("校准裁判需显式--real；将调用已配置API，讲解模型不会运行")
            from learning_agent.evaluation.calibration import calibrate

            result = asyncio.run(calibrate(args.samples, args.output, Settings()))
        else:
            from learning_agent.evaluation.legacy import import_legacy

            result = import_legacy(args.source, args.output)
    except (ValueError, OSError, LookupError, ModelError) as error:
        parser.error(str(error))
    # Content is returned only on explicit export; default output stays small.
    print(
        json.dumps(
            result
            if args.command != "regrade" and (args.command != "run" or not args.real)
            else {
                k: result[k]
                for k in ("planned", "counts", "reviewed", "reservations", "preparation_error")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
