"""Bounded paid RT3 acceptance; existing course assets, no automatic test retries."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--page", type=int, default=14)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--captions-json", type=Path, required=True)
    args = parser.parse_args()
    if not args.real:
        parser.error("需显式 --real；最多三个学习回合，会使用已配置API并产生费用")
    if args.output.exists():
        parser.error("输出目录必须是新目录")
    settings = Settings().model_copy(
        update={
            "data_root": args.output,
            "stream_answer": True,
            "max_steps": 4,
            "enable_context_summary": False,
            "embedding_provider": "disabled",
            "retrieval_mode": "bm25",
        }
    )
    if not settings.api_key.get_secret_value():
        parser.error("请先配置本机 LA_API_KEY")
    app = create_app(settings)
    store = app.state.store
    sid = store.space("RT3公开课程流式验收")["id"]
    pdf = store.add_pdf(sid, args.pdf.name, args.pdf.read_bytes())
    media = store.add_media_file(sid, args.media.name, args.media, args.media.suffix, move=False)
    shutil.copyfile(store.root / media["original_path"], store.root / media["preview_path"])
    with store.connect() as db:
        db.execute("UPDATE media SET state='ready' WHERE material_id=?", (media["material_id"],))
    captions = json.loads(args.captions_json.read_text())
    store.save_segments(
        sid,
        media["material_id"],
        captions["segments"],
        "local_asr:base:reused_course_acceptance",
    )
    cid = store.conversation(sid)["id"]
    cases = [
        (
            f"请实际查看PDF物理第{args.page}页图像，用三四句话解释图上的两条线各代表什么，以及为什么不能仅凭这张图断定哪条线预测更准。",
            {"material_id": pdf["id"], "page": args.page},
        ),
        ("我还是把简单模型理解成一定更准。请接着刚才的图，用一个简短例子纠正这个理解。", None),
        (
            "请同时核对这段视频20到75秒的转写与抽样画面，解释pre-norm中归一化与残差跳连的位置关系，给一个可回听的时间段；明确字幕或画面不能确认的部分，简短回答。",
            {"material_id": media["material_id"], "start_seconds": 20, "end_seconds": 75},
        ),
    ]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "protocol": settings.protocol,
        "conversation_id": cid,
        "turns": [],
    }
    with TestClient(app) as client:
        for question, focus in cases:
            started, submitted_at = time.monotonic(), datetime.now(UTC)
            payload = {"client_request_id": str(uuid4()), "content": question, "focus": focus}
            result = client.post(f"/api/conversations/{cid}/runs", json=payload)
            result.raise_for_status()
            rid = result.json()["run_id"]
            accepted = time.monotonic() - started
            assert (
                client.post(f"/api/conversations/{cid}/runs", json=payload).json()["run_id"] == rid
            )
            refreshed = False
            deadline = started + settings.timeout_seconds * 3 + 20
            while time.monotonic() < deadline:
                run = client.get(f"/api/runs/{rid}").json()
                if run.get("draft") and not refreshed:
                    snapshot = client.get(f"/api/conversations/{cid}/runs?limit=1").json()[0]
                    refreshed = (
                        snapshot["id"] == rid
                        and snapshot["draft"]["offset"] >= run["draft"]["offset"]
                    )
                if run["state"] not in {"queued", "running", "cancelling"}:
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError("验收等待超时，不自动重发")
            events = store.progress_events(rid, limit=1000)

            def latency(kind, events=events, submitted_at=submitted_at):
                event = next((e for e in events if e["type"] == kind), None)
                return (
                    round(
                        (
                            datetime.fromisoformat(event["created_at"]) - submitted_at
                        ).total_seconds(),
                        3,
                    )
                    if event
                    else None
                )

            with store.connect() as db:
                operations = [
                    dict(r)
                    for r in db.execute(
                        "SELECT purpose,logical_key,attempt,state,usage_status FROM run_operations WHERE run_id=? ORDER BY rowid",
                        (rid,),
                    )
                ]
            turn = {
                "run_id": rid,
                "question": question,
                "state": run["state"],
                "answer": run["answer"],
                "accepted_s": round(accepted, 3),
                "first_progress_s": latency("model.started"),
                "first_draft_s": latency("answer.delta"),
                "total_s": round(time.monotonic() - started, 3),
                "draft_refresh_ok": refreshed,
                "draft_events": sum(e["type"] == "answer.delta" for e in events),
                "stream_modes": [
                    e["payload"]["mode"] for e in events if e["type"] == "stream.mode"
                ],
                "operations": operations,
                "observations": store.run(rid)["events"],
            }
            report["turns"].append(turn)
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )
            print(
                json.dumps(
                    {
                        k: turn[k]
                        for k in (
                            "run_id",
                            "state",
                            "first_draft_s",
                            "total_s",
                            "draft_events",
                            "stream_modes",
                        )
                    }
                ),
                flush=True,
            )
            if run["state"] != "completed":
                raise RuntimeError("真实验收未完成；结果已保存，不自动重发")
    print(
        json.dumps({"report": str(args.output / "report.json"), "completed": len(report["turns"])})
    )


if __name__ == "__main__":
    main()
