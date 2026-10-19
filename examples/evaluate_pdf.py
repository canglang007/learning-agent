"""Opt-in paid PDF lesson check, keeping user data/reports outside source control."""

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
    parser.add_argument("--page", type=int, required=True, help="1-based physical page")
    parser.add_argument("--question", required=True)
    parser.add_argument("--followup")
    options = parser.parse_args()
    settings = Settings()
    if not settings.api_key.get_secret_value():
        raise SystemExit("请先在本机配置 LA_API_KEY。")
    root = settings.data_root / "evaluations" / datetime.now(UTC).strftime("pdf-%Y%m%dT%H%M%S%fZ")
    settings = settings.model_copy(update={"data_root": root})
    rows = []
    with TestClient(create_app(settings)) as client:
        space = client.post("/api/spaces", json={"content": "PDF联合讲解验收"}).json()["id"]
        uploaded = client.post(
            f"/api/spaces/{space}/materials",
            files={"file": (options.file.name, options.file.read_bytes(), "application/pdf")},
        )
        uploaded.raise_for_status()
        material = uploaded.json()["id"]
        conv = client.post(f"/api/spaces/{space}/conversations").json()["id"]
        for question in [options.question, options.followup]:
            if not question:
                continue
            started = time.monotonic()
            response = client.post(
                f"/api/conversations/{conv}/messages",
                json={
                    "content": question,
                    "focus": {"material_id": material, "page": options.page},
                },
            )
            response.raise_for_status()
            result = response.json()
            run = client.get(f"/api/runs/{result['run_id']}").json()
            row = {
                **result,
                "question": question,
                "events": run["events"],
                "latency_s": round(time.monotonic() - started, 2),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "state": row["state"],
                        "tools": [e["name"] for e in row["events"] if e["type"] == "tool"],
                        "latency_s": row["latency_s"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["state"] == "failed":
                break
    (root / "report.json").write_text(
        json.dumps(
            {
                "model": settings.model,
                "protocol": settings.protocol,
                "file": str(options.file),
                "physical_page": options.page,
                "quality_review": "pending",
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("报告：" + str(root / "report.json"), flush=True)


if __name__ == "__main__":
    main()
