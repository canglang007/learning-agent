"""生成 learning-core-v2 用例，并为多模态语料写入 corpus 资产映射（准备脚本）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from learning_agent.evaluation.ledger import Ledger
from learning_agent.retrieval import INDEX_VERSION
from learning_agent.store import Store

ALIASES = {
    "cs_pdf": "cs336-architecture.pdf",
    "cs_video": "cs336-prenorm.mp4",
    "zh_video": "ntu-tool-use.mp4",
    "silent_video": "cs336-prenorm-silent.mp4",
    "optimization": "optimization.pdf",
    "worksheet": "worksheet.pdf",
    "notes": "study-notes.md",
}
SOURCE_START = {"cs_video": 353, "zh_video": 3030}

suite = json.loads(Path("evals/learning-core-v1.json").read_text(encoding="utf-8"))
cases = {case["id"]: case for case in suite["cases"]}

cases["zh_chart"] = {
    "id": "zh_chart",
    "family": "zh_chart",
    "split": "dev",
    "tags": ["chinese", "chart", "image"],
    "settings": {"max_steps": 4, "enable_context_summary": False},
    "turns": [
        {
            "id": "explain",
            "conversation": "main",
            "input": {
                "question": (
                    "看第 51 页那张损失面示意图：当损失在一个方向变化很快、"
                    "另一个方向变化很慢时，梯度下降的路径会出什么问题？"
                ),
                "focus": {"asset": "optimization", "page": 51},
            },
            "gold": {
                "required_concepts": [
                    ("指出沿陡峭方向来回抖动、沿平缓方向前进极慢；根源是 Hessian 条件数过大")
                ],
                "evidence_targets": [{"asset": "optimization", "pages": [51], "modality": "image"}],
            },
            "actions": [],
        },
        {
            "id": "followup",
            "conversation": "main",
            "input": {"question": "为什么说这是条件数问题？判断依据是什么？"},
            "gold": {
                "required_concepts": [
                    "条件数是最大与最小曲率（奇异值）之比；图上等高线越扁，比值越大"
                ]
            },
            "actions": [],
        },
    ],
}

cases["silent_video"] = {
    "id": "silent_video",
    "family": "silent_video",
    "split": "dev",
    "tags": ["video", "no_caption", "image"],
    "settings": {"max_steps": 4, "enable_context_summary": False},
    "turns": [
        {
            "id": "explain",
            "conversation": "main",
            "input": {
                "question": (
                    "这段视频既没有声音也没有字幕。看 00:05–00:25 的画面，"
                    "它在展示什么？请把画面观察和你的推测分开写。"
                ),
                "focus": {"asset": "silent_video", "start_seconds": 0, "end_seconds": 30},
            },
            "gold": {
                "required_concepts": [
                    (
                        "描述画面里实际出现的架构示意/幻灯片内容；"
                        "明确无声无字幕时不能把推断当作讲者原话"
                    )
                ],
                "evidence_targets": [
                    {"asset": "silent_video", "seconds": [5, 25], "modality": "image"}
                ],
            },
            "actions": [],
        },
        {
            "id": "followup",
            "conversation": "main",
            "input": {"question": "如何让我回放最关键的时间点？"},
            "gold": {"required_concepts": ["给出可回放的起止秒数，并说明时间锚点来自媒体文件本身"]},
            "actions": [],
        },
    ],
}

limit = cases["video_limit"]
limit["turns"][0]["gold"]["forbidden_claims"] = [
    "给出 4060 上 pre-norm 相对 post-norm 的具体加速倍数（例如 1.5x、2x）",
    "把 FLOPs 比例当作实测加速比",
]

# 上面替换的是映射里的元素，必须写回列表，否则新用例不会进入评测集。
suite["cases"] = [cases[case["id"]] for case in suite["cases"]]

suite["id"] = "learning-core-v2"
suite["version"] = "2"
suite["supersedes"] = "1"
suite["change_note"] = (
    "相对 v1 的多模态评测修正：(1) silent_video 原先没有 focus、"
    "语料里也不存在无字幕视频，answer 全靠文字，evidence_targets 为空——"
    "v2 新增真实无声视频资产 cs336-prenorm-silent.mp4（从 cs336-prenorm.mp4 "
    "20-50 秒裁出、去除音轨、30 秒），该资产 chunks=0 文字检索不可达，"
    "只能 inspect_media 看帧，并要求图像帧落在 5-25 秒，未看图即失败；"
    "(2) zh_chart 原先没有 focus、问的是语料里不存在的训练/验证损失曲线，"
    "v2 改为 optimization.pdf 第 51 页真实损失面示意图，要求 modality=image；"
    "(3) video_limit 补 forbidden_claims，把“编造 4060 实测倍数”列入复核项。"
)
Path("evals/learning-core-v2.json").write_text(
    json.dumps(suite, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print("写入 evals/learning-core-v2.json，cases:", len(suite["cases"]))

root = Path("data/evaluations/multimodal-v1")
store = Store(root / "corpus")
space = store.spaces()[0]["id"]
materials = {m["name"]: m for m in store.materials(space)}
assets = {}
for alias, name in ALIASES.items():
    material = materials[name]
    assets[alias] = {
        "id": material["id"],
        "name": name,
        "format": material["format"],
        "source_start_seconds": SOURCE_START.get(alias, 0),
        "revision": material["revision"],
    }
ledger = Ledger(root / "experiment.db")
ledger.put(
    "corpus",
    {
        "space_id": space,
        "assets": assets,
        "index_version": INDEX_VERSION,
        "index_status": {
            "chunks": 208,
            "embedded": 208,
            "provider": "local",
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "state": "ready",
            "dense_ready": True,
        },
    },
)
print("写入 corpus 资产映射:", ", ".join(sorted(assets)))
