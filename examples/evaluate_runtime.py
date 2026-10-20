"""Small paid RT1 acceptance; use --real explicitly, results stay outside Git."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Allow calls to the configured model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.real:
        parser.error("真实模型验收需要显式添加 --real；离线验证请运行 tests/test_runtime.py")
    base = Settings()
    if not base.api_key.get_secret_value():
        parser.error("请先在本机配置 LA_API_KEY")
    root = args.output or base.data_root / "evaluations" / datetime.now(UTC).strftime(
        "runtime-rt1-%Y%m%d-%H%M%S"
    )
    if root.exists():
        parser.error("输出目录必须是新目录，避免改动既有资料")
    settings = base.model_copy(
        update={"data_root": root, "max_steps": 4, "enable_context_summary": False}
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "protocol": settings.protocol,
        "turns": [],
    }
    with TestClient(create_app(settings)) as client:
        sid = client.post("/api/spaces", json={"content": "RT1真实验收"}).json()["id"]
        cid = client.post(f"/api/spaces/{sid}/conversations").json()["id"]
        report.update(space_id=sid, conversation_id=cid)
        for question in [
            "请用两三句话解释梯度下降的学习率。",
            "我把学习率理解为每步走多远，请用一个简单数值例子解释，简短回答。",
        ]:
            payload = {"content": question, "client_request_id": str(uuid4())}
            started = time.monotonic()
            submitted = client.post(f"/api/conversations/{cid}/runs", json=payload)
            assert submitted.status_code == 202, submitted.text
            rid = submitted.json()["run_id"]
            accepted_s = round(time.monotonic() - started, 3)
            assert (
                client.post(f"/api/conversations/{cid}/runs", json=payload).json()["run_id"] == rid
            )
            assert client.get(f"/api/conversations/{cid}/runs?limit=1").json()[0]["id"] == rid
            deadline = time.monotonic() + settings.timeout_seconds * 2 + 10
            while time.monotonic() < deadline:
                current = client.get(f"/api/runs/{rid}").json()
                if current["state"] not in {"queued", "running", "cancelling"}:
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("运行未在验收期限内结束；不自动重试模型请求")
            events = client.get(f"/api/runs/{rid}/events").json()
            replay = client.get(
                f"/api/runs/{rid}/events?after=999",
                headers={"Accept": "text/event-stream", "Last-Event-ID": "1"},
            )
            with client.app.state.store.connect() as db:
                operations = [
                    dict(r)
                    for r in db.execute(
                        "SELECT purpose,step,state FROM run_operations WHERE run_id=?", (rid,)
                    )
                ]
            report["turns"].append(
                {
                    "run_id": rid,
                    "state": current["state"],
                    "question": question,
                    "answer": current["answer"],
                    "accepted_s": accepted_s,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "event_types": [e["type"] for e in events],
                    "sse_replay_ok": replay.status_code == 200
                    and "event: progress" in replay.text
                    and "id: 1\n" not in replay.text,
                    "operations": operations,
                }
            )
            (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {
                        "run_id": rid,
                        "state": current["state"],
                        "accepted_s": accepted_s,
                        "elapsed_s": report["turns"][-1]["elapsed_s"],
                    }
                ),
                flush=True,
            )
            if current["state"] != "completed":
                raise RuntimeError("本轮验收未完成；报告已保存，不自动重试")
    print(json.dumps({"report": str(root / "report.json"), "completed": len(report["turns"])}))


if __name__ == "__main__":
    main()
