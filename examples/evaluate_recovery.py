"""Paid RT2 acceptance: kill after the first durable model response, then recover."""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import profile_id
from learning_agent.execution_store import execution_profile
from learning_agent.store import Store


def settings_for(root):
    return Settings().model_copy(
        update={
            "data_root": root,
            "max_steps": 4,
            "enable_context_summary": False,
            "embedding_provider": "disabled",
            "retrieval_mode": "bm25",
        }
    )


async def run_worker(root, pause):
    settings = settings_for(root)
    app = create_app(settings)
    store = app.state.store
    original = store.complete_operation

    def complete(*args, **kwargs):
        result = original(*args, **kwargs)
        with store.connect() as db:
            operation = db.execute(
                "SELECT purpose,step FROM run_operations WHERE id=?", (args[2],)
            ).fetchone()
        if pause and operation["purpose"] == "main_model":
            (root / "response-saved").write_text(
                json.dumps({"operation_id": args[2], "step": operation["step"]})
            )
            while True:
                time.sleep(0.05)
        return result

    store.complete_operation = complete
    async with app.router.lifespan_context(app):
        await asyncio.Event().wait()


def spawn(root, pause=False):
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--real",
        "--worker",
        "--output",
        str(root.resolve()),
    ]
    if pause:
        args.append("--pause-after-response")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop(process, kill=False):
    if process.poll() is None:
        if kill:
            process.kill()
        else:
            process.terminate()
    process.communicate(timeout=5)


def operations(store, rid):
    with store.connect() as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT purpose,logical_key,attempt,state,usage_status,reserved_seconds,elapsed_seconds,image_reserved FROM run_operations WHERE run_id=? ORDER BY rowid",
                (rid,),
            )
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pause-after-response", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.real:
        parser.error("需显式 --real，运行将调用已配置的模型并产生费用")
    if args.worker:
        asyncio.run(run_worker(args.output, args.pause_after_response))
        return
    if not args.pdf or not args.pdf.is_file():
        parser.error("请提供已有真实课程PDF路径")
    root = args.output or Path("data/evaluations") / datetime.now(UTC).strftime(
        "runtime-rt2-%Y%m%d-%H%M%S"
    )
    if root.exists():
        parser.error("输出目录必须是新目录")
    settings = settings_for(root)
    if not settings.api_key.get_secret_value():
        parser.error("请先在本机配置 LA_API_KEY")
    store = Store(root)
    sid = store.space("RT2真实课程恢复验收")["id"]
    material = store.add_pdf(sid, args.pdf.name, args.pdf.read_bytes())
    mid = material["id"]
    store.page(sid, mid, args.page)
    cid = store.conversation(sid)["id"]
    question = f"请先实际查看这份PDF物理第{args.page}页的图像，再用两三句话讲清这页最主要的公式或图表；说明实际看到了什么，不补猜不可读内容。"
    rid = store.submit_run(
        cid,
        {
            "client_request_id": "real-recovery-0001",
            "content": question,
            "focus": {"material_id": mid, "page": args.page},
        },
        profile_id(settings),
        20,
        execution=execution_profile(settings),
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "protocol": settings.protocol,
        "run_id": rid,
        "conversation_id": cid,
        "material_id": mid,
        "page": args.page,
        "question": question,
    }
    process = spawn(root, True)
    started = time.monotonic()
    try:
        deadline = started + settings.timeout_seconds * 2 + 20
        while not (root / "response-saved").exists():
            if process.poll() is not None:
                raise RuntimeError("验收子进程提前退出")
            if store.run(rid)["state"] in {"failed", "interrupted"}:
                raise RuntimeError("首次请求未取得可用结果，不自动重发")
            if time.monotonic() > deadline:
                raise TimeoutError("等待首次响应超时，不自动重发")
            time.sleep(0.1)
        report["before_restart"] = {
            "operations": operations(store, rid),
            "phase": store.checkpoint(rid)["phase"],
        }
        print(json.dumps({"boundary": "response_saved", "run_id": rid}), flush=True)
    finally:
        stop(process, kill=True)
    process = spawn(root)
    try:
        deadline = time.monotonic() + settings.timeout_seconds * 2 + 20
        while store.run(rid)["state"] not in {"completed", "failed", "cancelled", "interrupted"}:
            if process.poll() is not None:
                raise RuntimeError("恢复子进程提前退出")
            if time.monotonic() > deadline:
                raise TimeoutError("恢复验收超时，不自动重试")
            time.sleep(0.1)
        final = store.public_run(rid)
        # A transient interrupted state can appear while startup queues a safe checkpoint.
        if final["state"] == "interrupted" and store.recovery_options(rid) == "resume":
            while time.monotonic() < deadline:
                time.sleep(0.1)
                final = store.public_run(rid)
                if final["state"] in {"completed", "failed", "cancelled"}:
                    break
        report.update(
            state=final["state"],
            answer=final["answer"],
            recovery_count=final["recovery_count"],
            elapsed_s=round(time.monotonic() - started, 3),
            operations=operations(store, rid),
            events=store.run(rid)["events"],
        )
        report["first_model_not_resent"] = (
            len([op for op in report["operations"] if op["logical_key"] == "main_model:1"]) == 1
        )
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "state": final["state"],
                    "first_model_not_resent": report["first_model_not_resent"],
                    "report": str(root / "report.json"),
                }
            ),
            flush=True,
        )
        if final["state"] != "completed" or not report["first_model_not_resent"]:
            raise RuntimeError("验收未完成；报告已保存，不自动重发")
    finally:
        stop(process)


if __name__ == "__main__":
    main()
