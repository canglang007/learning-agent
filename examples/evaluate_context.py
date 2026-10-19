"""Opt-in paid context check using a real PDF; outputs stay under data/."""

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
    parser.add_argument("--first-page", type=int, required=True)
    parser.add_argument("--second-page", type=int, required=True)
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    base = Settings()
    if not base.api_key.get_secret_value():
        raise SystemExit("请先配置 LA_API_KEY；不会输出密钥。")
    root = options.output or base.data_root / "evaluations" / datetime.now(UTC).strftime(
        "context-%Y%m%dT%H%M%S%fZ"
    )
    if root.exists() and any(root.iterdir()):
        raise SystemExit("输出目录须为空，避免覆盖既有验收。")
    settings = Settings(
        _env_file=None,
        **{
            **base.model_dump(),
            "data_root": root,
            "context_budget_tokens": options.budget,
            "max_images_per_turn": 1,
            "max_context_images": 1,
        },
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "protocol": settings.protocol,
        "budget": options.budget,
        "turns": [],
        "review": "工程路径与usage记录；教学正确性需另行对照原页复核。",
    }
    pages = [options.first_page, options.first_page, options.second_page, options.first_page]
    questions = [
        (
            f"请实际查看第{pages[0]}页的页面图像，解释这页最重要的图示或公式，以及它想说明什么。"
            "只围绕实际页面讲，简洁说明关键关系和条件。"
        ),
        "刚才的解释我还没理解，请沿用同一页，用一个具体例子解释其中一个重要关系，指出适用条件。",
        f"请实际查看第{pages[2]}页的页面图像，概括它与前页的联系与区别。若内容不连贯，明确指出。两三段即可。",
        (
            f"回到第{pages[3]}页，请重新看图核对：上一轮的概括是否适用于它？"
            f"先说明是否需要修正，再解释我应该怎么理解；不要照搬第{pages[2]}页的结论。"
        ),
    ]
    with TestClient(create_app(settings)) as client:
        space = client.post("/api/spaces", json={"content": "课程上下文连续验收"}).json()["id"]
        uploaded = client.post(
            f"/api/spaces/{space}/materials",
            files={"file": (options.file.name, options.file.read_bytes(), "application/pdf")},
        )
        uploaded.raise_for_status()
        material = uploaded.json()["id"]
        for page in set(pages):
            client.get(f"/api/spaces/{space}/materials/{material}/pages/{page}").raise_for_status()
        conversation = client.post(f"/api/spaces/{space}/conversations").json()["id"]
        report.update(space_id=space, conversation_id=conversation, material_id=material)
        for question, page in zip(questions, pages):
            started = time.monotonic()
            response = client.post(
                f"/api/conversations/{conversation}/messages",
                json={"content": question, "focus": {"material_id": material, "page": page}},
            )
            response.raise_for_status()
            result = response.json()
            run = client.get(f"/api/runs/{result['run_id']}").json()
            report["turns"].append(
                {
                    **result,
                    "question": question,
                    "page": page,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "run": run,
                }
            )
            snapshots = [s for t in report["turns"] for s in t["run"]["context"]]
            report["summary"] = {
                "completed_turns": sum(t["state"] == "completed" for t in report["turns"]),
                "model_returns": sum(
                    e["type"] == "model" for t in report["turns"] for e in t["run"]["events"]
                ),
                "image_removal_triggered": any(
                    o["reason"]
                    in {"image_limit", "image_budget_pressure", "image_already_attached"}
                    for s in snapshots
                    for o in s.get("omissions", [])
                ),
                "history_reduction_triggered": any(
                    o["reason"] in {"tool_budget_pressure", "history_budget_pressure"}
                    for s in snapshots
                    for o in s.get("omissions", [])
                ),
            }
            (root / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "turn": len(report["turns"]),
                        "state": result["state"],
                        "steps": len(run["context"]),
                        "context_limited": result["context_limited"],
                        "elapsed_s": report["turns"][-1]["elapsed_s"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result["state"] == "failed":
                break
    print(
        json.dumps({"report": str(root / "report.json"), **report["summary"]}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
