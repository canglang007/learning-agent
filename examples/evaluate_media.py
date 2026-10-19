"""Opt-in ASR and paid model check using actual decoded media; reports stay local."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--companion", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--question", required=True)
    parser.add_argument("--followup")
    options = parser.parse_args()
    settings = Settings()
    if not settings.api_key.get_secret_value():
        raise SystemExit("请先配置 LA_API_KEY")
    root = settings.data_root / "evaluations" / datetime.now(UTC).strftime("media-%Y%m%dT%H%M%S%fZ")
    settings = settings.model_copy(update={"data_root": root})
    report = {
        "file": str(options.file),
        "model": settings.model,
        "protocol": settings.protocol,
        "quality_review": "pending",
        "cases": [],
    }
    with TestClient(create_app(settings)) as client:
        space = client.post("/api/spaces", json={"content": "音视频联合讲解验收"}).json()["id"]

        def upload(path):
            response = client.post(
                f"/api/spaces/{space}/materials", files={"file": (path.name, path.read_bytes())}
            )
            response.raise_for_status()
            return response.json()

        material = upload(options.file)["id"]
        if options.companion:
            upload(options.companion)
        base = f"/api/spaces/{space}/materials/{material}"
        if options.captions:
            response = client.post(
                base + "/captions",
                files={"file": (options.captions.name, options.captions.read_bytes())},
            )
            response.raise_for_status()
        if options.transcribe:
            start = time.monotonic()
            response = client.post(base + "/transcribe")
            response.raise_for_status()
            report["asr_latency_s"] = round(time.monotonic() - start, 2)
        report["media"] = client.get(base + "/media").json()
        report["segments"] = client.get(base + "/segments").json()
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "media_state": report["media"]["state"],
                    "transcript_state": report["media"]["transcript_state"],
                    "asr_s": report.get("asr_latency_s"),
                    "asr_error": report["media"]["transcript_error"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if options.transcribe and report["media"]["transcript_state"] != "ready":
            print("转写未完成；报告已保存，不继续消耗讲解API。", flush=True)
            return
        conv = client.post(f"/api/spaces/{space}/conversations").json()["id"]
        for question in [options.question, options.followup]:
            if not question:
                continue
            start = time.monotonic()
            result = client.post(
                f"/api/conversations/{conv}/messages",
                json={
                    "content": question,
                    "focus": {
                        "material_id": material,
                        "start_seconds": 0,
                        "end_seconds": min(120, report["media"]["duration"]),
                    },
                },
            )
            result.raise_for_status()
            row = result.json()
            row.update(
                {
                    "question": question,
                    "latency_s": round(time.monotonic() - start, 2),
                    "events": client.get(f"/api/runs/{row['run_id']}").json()["events"],
                }
            )
            report["cases"].append(row)
            (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {
                        "state": row["state"],
                        "latency_s": row["latency_s"],
                        "tools": [e["name"] for e in row["events"] if e["type"] == "tool"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["state"] == "failed":
                break
    print("报告：" + str(root / "report.json"), flush=True)


if __name__ == "__main__":
    main()
