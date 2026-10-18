"""Opt-in, paid live text-agent check. Reports are local data, never credentials."""

import json
import time
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings

LESSON = """# 近似与忽略高阶项：人工编写的验收笔记
这是一份用于检查 Agent 行为的短材料，不是课程原文，也不是完整评测集。
近似不是严格相等，需要说明误差与适用范围。
例：F(x)=1+x+x²。若 x=0.01，x²=0.0001，比保留的 1+x 小很多。
若 x=2，x²=4，不能按同样理由省略。
某项绝对值小不代表一定能省略；要与保留的项比较，并考虑任务允许的误差。
例：0.001+0.0001，后者虽然绝对值小，却是前者的十分之一。
用户问到具体推导时，应明确当前表达式、近似区域和误差要求。
"""


def main():
    settings = Settings()
    if not settings.api_key.get_secret_value():
        raise SystemExit("请先在本机 .env 配置 LA_API_KEY。")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = settings.data_root / "evaluations" / stamp
    settings = settings.model_copy(update={"data_root": root})
    rows = []
    with TestClient(create_app(settings)) as client:
        space = client.post("/api/spaces", json={"content": "文字学习真实调用验收"}).json()
        conv = client.post(f"/api/spaces/{space['id']}/conversations").json()

        def ask(name, question):
            started = time.monotonic()
            response = client.post(
                f"/api/conversations/{conv['id']}/messages", json={"content": question}
            )
            response.raise_for_status()
            result = response.json()
            run = client.get(f"/api/runs/{result['run_id']}").json()
            row = {
                "name": name,
                "question": question,
                **result,
                "latency_s": round(time.monotonic() - started, 2),
                "events": run["events"],
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "name": name,
                        "state": result["state"],
                        "tools": [e["name"] for e in run["events"] if e["type"] == "tool"],
                        "latency_s": row["latency_s"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return result["state"] != "failed"

        if ask(
            "general_explanation",
            "请用一个很短的生活例子，解释近似等于与严格相等的区别。这是通用问题，不需要资料。",
        ):
            upload = client.post(
                f"/api/spaces/{space['id']}/materials",
                files={"file": ("approximation-notes.md", LESSON.encode(), "text/markdown")},
            )
            upload.raise_for_status()
            if ask(
                "consult_material",
                "我刚上传了一份近似笔记。请先查阅它，再解释为什么不能因为某一项绝对值很小就直接忽略，结合笔记里的具体数字。",
            ):
                ask(
                    "adapt_explanation",
                    "我还是没懂怎么比较，请换成花钱买东西的例子，讲得直观一点。不要从头复述笔记，也不要问我一串测验题。",
                )
        messages = client.get(f"/api/conversations/{conv['id']}/messages").json()
    report = {
        "model": settings.model,
        "protocol": settings.protocol,
        "material_type": "authored_test_note",
        "quality_review": "pending_human_review",
        "cases": rows,
        "messages": messages,
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告位置：" + str(report_path), flush=True)


if __name__ == "__main__":
    main()
