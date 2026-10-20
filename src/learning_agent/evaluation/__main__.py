import argparse
import asyncio
import json
from pathlib import Path

from learning_agent.config import Settings
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
        elif args.command == "retrieval":
            if not args.real:
                parser.error("检索组件可能调用向量服务，需要显式--real；不会调用讲解模型")
            from learning_agent.evaluation.retrieval import evaluate_retrieval

            result = evaluate_retrieval(
                args.experiment, json.loads(args.queries.read_text())["queries"], Settings()
            )
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
