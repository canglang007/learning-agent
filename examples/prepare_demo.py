"""Create a self-contained demo workspace with synthetic materials; no model calls."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from reportlab.pdfgen import canvas

from learning_agent.config import Settings
from learning_agent.retrieval import Retriever
from learning_agent.store import Store

NOTES = """# 合成矩阵演示笔记

矩阵乘法 C = A @ B 时，C 的第 i 行第 j 列等于 A 的第 i 行与 B 的第 j 列的对应元素乘积之和。

A = [[1, 2, 3], [4, 5, 6]]
B = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]

C[1,2] = 1×0 + 2×1 + 3×0 = 2。

注意：矩阵乘法不是对应元素直接相乘；维度需要满足 A 的列数等于 B 的行数。
这条笔记是合成演示数据，不是课程原文。
"""

QUESTION = "矩阵乘法里 C[1,2] 是怎么算出来的？"
ANSWER = (
    "C[1,2] 是 A 的第 1 行与 B 的第 2 列对应元素相乘再相加："
    "1×0 + 2×1 + 3×0 = 2。\n\n"
    "演示数据中的固定示例回答，用来说明页面结构；不是本次模型生成。"
)


def _worksheet_pdf() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setFont("Helvetica", 14)
    lines = [
        "Synthetic matrix worksheet (not course material)",
        "C = A @ B",
        "C[1,2] = row 1 of A dot column 2 of B",
        "A = [[1,2,3],[4,5,6]]",
        "B column 2 = [0,1,0]",
        "C[1,2] = 1*0 + 2*1 + 3*0 = 2",
    ]
    for index, line in enumerate(lines):
        pdf.drawString(72, 740 - index * 28, line)
    pdf.save()
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成自包含的 Learning Agent 演示工作区")
    parser.add_argument("--output", type=Path, default=Path("data/demo-public"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("演示目录必须是新目录，避免覆盖已有资料")
    root.mkdir(parents=True)

    store = Store(root)
    space = store.space("演示课程")
    note = store.add_material(space["id"], "matrix-notes.md", NOTES)
    pdf = store.add_pdf(space["id"], "matrix-worksheet.pdf", _worksheet_pdf())
    config = Settings(
        _env_file=None,
        data_root=root,
        retrieval_mode="bm25",
        embedding_provider="disabled",
    )
    Retriever(store, config).index.chunks(space["id"])

    conversation = store.conversation(space["id"])
    run_id = store.start_run(conversation["id"], QUESTION)
    groups = [
        {"id": run_id + ":input", "kind": "input", "messages": [{"role": "user", "content": QUESTION}]},
        {"id": run_id + ":answer", "kind": "answer", "messages": [{"role": "assistant", "content": ANSWER}]},
    ]
    store.finish_run(
        run_id,
        "completed",
        ANSWER,
        [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": ANSWER},
        ],
        archive={
            "profile": {"service": "demo", "host": "demo", "model": "demo", "protocol": "demo"},
            "input": {"question": QUESTION},
            "groups": groups,
        },
    )
    print(
        json.dumps(
            {
                "workspace": str(root),
                "space_id": space["id"],
                "conversation_id": conversation["id"],
                "materials": [note["name"], pdf["name"]],
                "model_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
