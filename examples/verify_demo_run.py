"""Run one real demo question through the local API and export evidence JSON.

Everything here goes through the same HTTP endpoints the web page uses, so the
record matches what a viewer can see: route, tool sequence, verified citation
positions and the per-step context estimate. It never reads the database
directly and never prints the API key.

The output stays under data/ and is not committed. Course material is not
copied into the report, only names, positions and counts.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_QUESTION = (
    "请实际查看cs336-architecture.pdf第10页，并结合cs336-prenorm.mp4的20到75秒，"
    "讲清pre-norm和post-norm的残差直通区别。三四句话即可，给一个可回听窗口。"
)
TERMINAL = {"completed", "budget_exhausted", "failed", "cancelled", "interrupted"}


def get(client, path, **kwargs):
    response = client.get(path, **kwargs)
    response.raise_for_status()
    return response.json()


def pick_space(client, wanted):
    spaces = get(client, "/api/spaces")
    if not spaces:
        raise SystemExit("没有资料空间，请先准备演示工作区")
    if not wanted:
        return spaces[0]
    for space in spaces:
        if space["id"] == wanted or wanted in space["name"]:
            return space
    names = ", ".join(f"{s['name']}（{s['id']}）" for s in spaces)
    raise SystemExit(f"找不到空间 {wanted}；现有：{names}")


def summarize_events(events):
    """Tool sequence with failure flags, plus the chosen route."""
    route, steps, failures = None, [], 0
    for event in events:
        payload = event.get("payload") or {}
        if event["type"] == "route.selected":
            route = {"route": payload.get("route"), "reason": payload.get("reason")}
        if event["type"] == "tool.finished":
            failures += bool(payload.get("error"))
            steps.append(
                {
                    "name": payload.get("name"),
                    "error": bool(payload.get("error")),
                    **{
                        key: payload[key]
                        for key in ("material_id", "page", "start_seconds", "end_seconds")
                        if key in payload
                    },
                }
            )
    return route, steps, failures


def summarize_context(snapshots):
    return [
        {
            "step": snapshot["step"],
            "estimate": snapshot.get("estimate"),
            "input_budget": snapshot.get("input_budget"),
            "omissions": [item.get("reason") for item in snapshot.get("omissions", [])],
            "selection": snapshot.get("selection"),
            "result": snapshot.get("result"),
        }
        for snapshot in snapshots
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8320")
    parser.add_argument("--space", default="", help="空间 id 或名称片段；缺省取第一个")
    parser.add_argument("--conversation", default="", help="复用已有会话；缺省新建")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument(
        "--access-code",
        default="",
        help="对外试用设了 LA_ACCESS_TOKEN 时填口令，脚本会先走一次登录",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/demo") / "multimodal-run-latest.json"
    )
    options = parser.parse_args()

    with httpx.Client(base_url=options.base_url, timeout=30) as client:
        if options.access_code:
            login = client.post("/login", data={"code": options.access_code, "next": "/"})
            if login.status_code not in (200, 303) or not client.cookies.get("la_access"):
                raise SystemExit(f"登录失败：{login.status_code}；口令错了就换个口令再试")
        status = get(client, "/api/status")
        if not status.get("configured"):
            raise SystemExit("模型未配置；本脚本只做真实运行，不生成占位记录")
        space = pick_space(client, options.space)
        materials = get(client, f"/api/spaces/{space['id']}/materials")
        conversation = (
            {"id": options.conversation}
            if options.conversation
            else client.post(f"/api/spaces/{space['id']}/conversations").json()
        )
        request_id = datetime.now(UTC).strftime("verify-%Y%m%dT%H%M%S%fZ")
        started = client.post(
            f"/api/conversations/{conversation['id']}/runs",
            json={"content": options.question, "client_request_id": request_id},
        )
        started.raise_for_status()
        run_id = started.json()["run_id"]
        print(f"已提交：{run_id}（真实模型调用，会计费）")

        deadline = time.monotonic() + options.timeout
        run = get(client, f"/api/runs/{run_id}")
        while run["state"] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(1.5)
            run = get(client, f"/api/runs/{run_id}")
        if run["state"] not in TERMINAL:
            raise SystemExit(f"等待超时：{run_id} 仍在 {run['state']}")

        events = get(client, f"/api/runs/{run_id}/events", params={"after": 0})
        route, steps, failures = summarize_events(events)
        context = summarize_context(run.get("context", []))
        last = max(context, key=lambda item: item["step"], default={})
        report = {
            "verified_at": datetime.now(UTC).isoformat(),
            "via": options.base_url,
            "workspace_space": {"id": space["id"], "name": space["name"]},
            "materials": [
                {
                    key: item[key]
                    for key in ("id", "name", "format", "page_count", "media_state")
                    if key in item
                }
                for item in materials
            ],
            "conversation_id": conversation["id"],
            "run_id": run_id,
            "model": status.get("model"),
            "protocol": status.get("protocol"),
            "route": route,
            "state": run["state"],
            "question": options.question,
            "answer": run.get("answer"),
            "references": run.get("references", []),
            "tool_steps": steps,
            "tool_failures": failures,
            "context_steps": context,
            "last_step_estimate": last.get("estimate"),
            "note": "本地真实运行记录；课程素材与数据库不随仓库分发。",
        }
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    verified = [ref for ref in report["references"] if ref.get("position_verified")]
    print(f"状态：{run['state']}；路由：{(route or {}).get('route')}")
    print(f"工具调用：{len(steps)} 次，失败 {failures} 次")
    print("序列：" + " → ".join(step["name"] for step in steps))
    print(f"已验证引用：{len(verified)}/{len(report['references'])}")
    print(f"最后一步图像数：{(report['last_step_estimate'] or {}).get('images')}")
    print(f"证据写入：{options.output}（本地，不提交）")


if __name__ == "__main__":
    main()
