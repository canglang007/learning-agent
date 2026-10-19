"""Opt-in paid long-conversation/state acceptance; raw artifacts stay local."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings

QUESTIONS = [
    (
        14,
        "我的学习目标是分清正则化、模型复杂度和泛化。请实际看第14页图像，解释它支持什么，不能证明什么。",
    ),
    (14, "先纠正一个用词：我关心的是函数曲线的复杂程度，不要把复杂模型直接说成参数一定更多。"),
    (14, "用两组明确是假设的数字，解释拟合误差加复杂度惩罚后，为什么模型选择可能改变。"),
    (14, "惩罚加大一定更好吗？给一个反例，先说明成立条件。"),
    (14, "我来复述：加入惩罚后总分更低，就证明测试误差也更低。请判断这句话的缺口。"),
    (14, "第一页只是在说明偏好简单，这一点明白了；我还没有理解整章。"),
    (15, "请实际查看第15页图像，和前页比较。灰色点代表什么，哪些身份只是可能解释？"),
    (15, "补充我的要求：灰色点没有明确标签时，不要把它直接当成已标注的测试集。"),
    (14, "重新看第14页，哪些刚才第15页的说法不能直接照搬？"),
    (14, "核对最早一轮我的学习目标，再用一句话说明目前还缺少哪类证据才能判断泛化。"),
    (14, "假如真实函数本来弯曲，为什么偏爱直线也可能失败？不要假设这张图已经证明真实函数。"),
    (14, "按我刚修正的目标，说明正则化强度应怎样评估，而不是凭训练误差选。"),
    (14, "只看训练与验证误差的差距就足够了吗？用一个欠拟合的例子反驳。"),
    (None, "换个主题讲梯度下降。先区分参数更新的方向与学习率大小，不用延伸到优化器列表。"),
    (None, "之前第14页那一点我明白了，我们现在仍然在讨论梯度下降。"),
    (None, "这里我的卡点是更新方向，不是学习率大小。请针对这个纠正解释。"),
    (None, "设一维梯度为2、学习率0.1、原参数1，请算一步更新并解释负号。"),
    (None, "我来复述：梯度为正，说明应该增加参数才能减小损失。哪里错了？"),
    (None, "为什么这个局部方向解释不等于保证很大的步长也下降？"),
    (None, "请读取第1轮用户原话，逐字核对我的最初学习目标；再说明它与中途修改后的目标有什么区别。"),
    (14, "暂时回看第14页，请实际看图说明模型选择，与刚才梯度更新分别解决什么问题。"),
    (14, "不要把我说旧图的某一点明白了当作我已经掌握整个正则化或优化。你还应保留什么不确定性？"),
    (14, "我的目标中途改过，请按当前目标举一个训练误差较低却不能选它的简短例子。"),
    (None, "围绕当前目标，指出还有哪些具体问题值得继续解释，不要给掌握分数或强制出题。"),
    (None, "验证误差偶然低一次能证明模型更好吗？补充实验波动的限制。"),
    (None, "参数更新方法与正则化可以同时使用吗？保持之前二者作用的区别。"),
    (None, "请核对我关于灰色点标签的原话，避免从历史概括中多推出测试集身份。"),
    (None, "讲法仍偏抽象，请用两个候选模型和独立验证集解释选择过程。"),
    (None, "这个例子里哪些是构造数字，哪些需要实际数据才有结论？"),
    (None, "用当前学习目标组织最后的解释，保留尚未解决的问题，先不要安排新的学习计划。"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--turns", type=int, choices=range(1, 31), default=24)
    parser.add_argument("--budget", type=int, default=40960)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    base = Settings()
    if not base.api_key.get_secret_value():
        raise SystemExit("请先配置 LA_API_KEY。此脚本调用模型并计费，不输出密钥。")
    root = options.output or base.data_root / "evaluations" / datetime.now(UTC).strftime(
        "learning-context-%Y%m%dT%H%M%S%fZ"
    )
    if root.exists() and any(root.iterdir()):
        raise SystemExit("请使用新的输出目录，避免覆盖报告。")
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
    }
    with TestClient(create_app(settings)) as client:
        space = client.post("/api/spaces", json={"content": "长对话学习状态验收"}).json()["id"]
        uploaded = client.post(
            f"/api/spaces/{space}/materials",
            files={"file": (options.file.name, options.file.read_bytes(), "application/pdf")},
        )
        uploaded.raise_for_status()
        material = uploaded.json()["id"]
        conv = client.post(f"/api/spaces/{space}/conversations").json()["id"]
        report.update(space_id=space, conversation_id=conv, material_id=material)
        endpoint = f"/api/conversations/{conv}"
        for index, (page, question) in enumerate(QUESTIONS[: options.turns], 1):
            if index == 12:
                before = client.get(endpoint + "/learning").json()
                response = client.patch(
                    endpoint + "/learning",
                    json={
                        "revision": before["revision"],
                        "goal": "根据独立验证证据选择正则化强度，不用训练误差代替泛化证据",
                    },
                )
                response.raise_for_status()
                report["user_correction"] = {
                    "before_revision": before["revision"],
                    "after": response.json(),
                }
            payload = {"content": question + " 请优先在150个中文字内讲清楚，必要时可保留公式。"}
            if page is not None:
                payload["focus"] = {"material_id": material, "page": page}
            if index in {6, 15}:
                payload["feedback"] = {
                    "kind": "understood",
                    "target_run_id": report["turns"][0]["run_id"],
                }
            started = time.monotonic()
            response = client.post(endpoint + "/messages", json=payload)
            response.raise_for_status()
            result = response.json()
            run = client.get("/api/runs/" + result["run_id"]).json()
            state = client.get(endpoint + "/learning").json()
            report["turns"].append(
                {
                    **result,
                    "question": question,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "run": run,
                    "learning": state,
                }
            )
            with client.app.state.store.connect() as db:
                summaries = [
                    dict(r)
                    for r in db.execute(
                        "SELECT id,covered_end,status,state_revision,result FROM context_summaries ORDER BY rowid"
                    )
                ]
            report["summary_attempts"] = summaries
            (root / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "turn": index,
                        "state": result["state"],
                        "revision": state["revision"],
                        "summary_attempts": len(summaries),
                        "elapsed_s": report["turns"][-1]["elapsed_s"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result["state"] == "failed":
                break
    print(
        json.dumps(
            {
                "report": str(root / "report.json"),
                "completed": sum(t["state"] == "completed" for t in report["turns"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
