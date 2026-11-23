"""从零构建整讲课程评测语料：五讲完整原片（两门课）+ 三份 PDF + 一份笔记。

和 ``prepare_multimodal_corpus.py`` 的区别：那份脚本是往既有基线里补一个无声视频，
这份换掉整个基线。原来的两个视频资产是两三分钟的切片（198 秒 / 270 秒），
切出来的分块只有 3 个，视频检索实际上是在三选一；这里换成整讲，分块数量级跟着
句子密度走，检索才谈得上"从几百段里找回那一段"。

语料只放**被题目用到的材料**。无声视频 `cs336-prenorm-silent.mp4` 不在这份语料里：
它在检索题里没有任何目标（0 个分块、0 条题），放进来只会让"语料里有它"看起来像
某种证据。它属于能力用例的输入（`evals/learning-core-v4.json` 的 `silent_video`），
由 `bindings.local.json` 单独提供给那条链路。

五讲的字幕来源刻意不统一，这本身就是对照：

- NTU（中文）、李宏毅两讲（中文）有官方 .vtt，直接解析，零推理成本；
- CS336 2025 一讲、2026 三讲（英文）都没有字幕，走产品自己的 ASR 任务
  （faster-whisper base、CPU int8、300 秒分片 + 5 秒重叠），所以这份语料同时是
  转写链路的实测记录。五讲都用同一个 ASR 配置，避免把转写质量差异混进检索难度。

用法：
    uv sync --extra embed --extra asr
    python scripts/build_lecture_corpus.py --root data/evaluations/multimodal-v1/corpus

原片下载（``data/acceptance/real-courses/`` 下，不入库、不随仓库分发）::

    NODE=$(command -v node)
    yt-dlp --js-runtimes "node:$NODE" \\
      --write-subs --sub-langs "zh-TW" --convert-subs vtt \\
      -o "data/acceptance/real-courses/<dir>/%(id)s.%(ext)s" "<url>"

选流的原则是**处理成本最小**，不是"产品只吃什么"：默认 ``bv*[height<=720]+ba``，
拿到 webm/opus 也照样入库（产品按容器+编码分级：能播就原样提供、只差壳就 remux）。
``--js-runtimes`` 不是可选项：缺 JS 运行时拿不到正常签名的流，实测被限速到
51 KB/s，指定 node 后 518 KB/s。

脚本可续跑：按材料名跳过已入库的，加新材料不用重跑已有转写。目标目录非空且没加
``--force`` 时会续跑；``--force`` 只清掉 materials/ 与 workspace.db，原始视频仍在
``data/acceptance/real-courses/`` 下。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from learning_agent.config import Settings
from learning_agent.jobs import JobConflict, Jobs
from learning_agent.media import parse_captions
from learning_agent.retrieval import Embedder, Retriever
from learning_agent.store import Store
from learning_agent.worker import Worker

ACCEPT = Path("data/acceptance")

PDFS = [
    ("cs336-architecture.pdf", ACCEPT / "real-courses/cs336-architecture.pdf"),
    ("optimization.pdf", ACCEPT / "optimization.pdf"),
    ("worksheet.pdf", Path("data/demo/worksheet.pdf")),
]
NOTES = [("study-notes.md", Path("data/demo/study-notes.md"))]
LECTURES = [
    {
        "name": "cs336-lecture.mp4",
        "path": ACCEPT / "real-courses/cs336-original.mp4",
        "url": "https://www.youtube.com/watch?v=ptFiH_bHnJw",
        "captions": None,
    },
    {
        "name": "ntu-lecture.mp4",
        "path": ACCEPT / "real-courses/ntu-original.mp4",
        "url": "https://www.youtube.com/watch?v=M2Yg1kwPpts",
        "captions": ACCEPT / "real-courses/ntu-source.zh-TW.vtt",
    },
    # 2026 版 CS336 三讲：同一门课的不同主题（注意力替代方案、scaling、推理）。
    # 加它们的理由不是"材料越多越好"，而是原来只有一门课的一讲，检索模型可能
    # 只学到那一讲的措辞；跨主题才能看出检索是在读语义还是在匹配表面词。
    {
        "name": "cs336-lecture4.mp4",
        "path": ACCEPT / "real-courses/cs336-2026/cKSwj_qZ8Jg.mp4",
        "url": "https://www.youtube.com/watch?v=cKSwj_qZ8Jg",
        "captions": None,
    },
    {
        "name": "cs336-lecture9.mp4",
        "path": ACCEPT / "real-courses/cs336-2026/Q15rhEWZPQ4.mp4",
        "url": "https://www.youtube.com/watch?v=Q15rhEWZPQ4",
        "captions": None,
    },
    {
        "name": "cs336-lecture10.mp4",
        "path": ACCEPT / "real-courses/cs336-2026/EfM546A79aM.mp4",
        "url": "https://www.youtube.com/watch?v=EfM546A79aM",
        "captions": None,
    },
    # 李宏毅这两讲是同一主题的上下篇（Flash Attention / KV Cache），官方繁体字幕。
    # 跨讲对照类题目（"两种加速各自优化什么"）只有在多讲语料里才出得来。
    {
        "name": "hungyi-flash-attention.mp4",
        "path": ACCEPT / "real-courses/hungyi-2025/vXb2QYOUzl4.mp4",
        "url": "https://www.youtube.com/watch?v=vXb2QYOUzl4",
        "captions": ACCEPT / "real-courses/hungyi-2025/vXb2QYOUzl4.zh-TW.vtt",
    },
    {
        "name": "hungyi-kv-cache.mp4",
        "path": ACCEPT / "real-courses/hungyi-2025/fDQaadKysSA.mp4",
        "url": "https://www.youtube.com/watch?v=fDQaadKysSA",
        "captions": ACCEPT / "real-courses/hungyi-2025/fDQaadKysSA.zh-TW.vtt",
    },
]


def main():
    parser = argparse.ArgumentParser(description="构建整讲课程评测语料")
    parser.add_argument("--root", default="data/evaluations/multimodal-v1/corpus")
    parser.add_argument("--force", action="store_true", help="清空目标目录后重建")
    args = parser.parse_args()

    root = Path(args.root)
    if args.force and root.exists():
        import shutil

        for item in root.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    elif root.exists() and any(root.iterdir()):
        print(f"{root} 已存在，按已有材料续跑（要重建请加 --force）")

    settings = Settings(
        _env_file=None,
        data_root=root,
        retrieval_mode="hybrid",
        # 造语料这一步显式用进程内模型：不需要 ollama 守护进程，结果可复现。
        # 这里产出的向量不会被评测用到——评测脚本复制语料后会按自己的模型
        # （默认 bge-m3）整批重建，encoder 指纹不同就判定为待嵌入。
        embedding_provider="local",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # 复用共享缓存：不指定时它会在 data_root 下重下一份，几十兆到上百兆，
        # 而且一旦网络卡住整个索引就停在那儿，看上去像在算其实在等下载。
        embedding_cache_dir=str(Path.home() / ".cache" / "la-models"),
    )
    store = Store(root)
    space = (store.spaces() or [store.space("评测资料基线")])[0]["id"]
    print(f"space {space}  root {root}")

    # 可续跑：按名字跳过已入库的材料。转写一次要几分钟，重跑不该再付一遍。
    existing = {m["name"] for m in store.materials(space)}
    for name, path in PDFS:
        if name in existing:
            print(f"  pdf   {name:26s} 已存在，跳过")
            continue
        item = store.add_pdf(space, name, path.read_bytes(), settings.ocr_min_text_characters)
        print(f"  pdf   {name:26s} pages={item['page_count']} ocr_pending={item['ocr_pending']}")

    for name, path in NOTES:
        if name in existing:
            print(f"  note  {name} 已存在，跳过")
            continue
        store.add_material(space, name, path.read_text(encoding="utf-8"))
        print(f"  note  {name}")

    jobs = Jobs(store)
    for lecture in LECTURES:
        if lecture["name"] in existing:
            print(f"  video {lecture['name']:26s} 已存在，跳过")
            continue
        # move=False：原片留在 data/acceptance 作为证据，语料留自己的副本。
        item = store.add_media_file(space, lecture["name"], lecture["path"], ".mp4", move=False)
        material = item["material_id"]
        print(f"  video {lecture['name']:26s} duration={item['duration']:.1f}s state={item['state']}")
        jobs.enqueue("preview", space, settings, material)
        started = time.time()
        if lecture["captions"]:
            info = store.media(space, material)
            raw = lecture["captions"].read_bytes()
            segments = parse_captions(raw, info["duration"])
            store.save_segments(
                space, material, segments, "uploaded:" + lecture["captions"].name
            )
            print(f"        captions -> {len(segments)} segments ({time.time() - started:.1f}s)")
        else:
            jobs.enqueue("transcribe", space, settings, material)
            print("        no captions -> queued transcribe (ASR)")
        store.set_source(space, material, {"url": lecture["url"], "start_seconds": 0})

    print("worker drain ...")
    drain_started = time.time()
    with Worker(settings) as worker:
        worker.drain()
    print(f"worker finished in {time.time() - drain_started:.1f}s")

    # 上一轮被中断的索引任务会被 recover 标成 paused，而 enqueue 对同版本同配置
    # 的 paused 任务会原样返回，于是永远不会再跑。先撤掉再重新排一次。
    with store.connect() as db:
        db.execute(
            "UPDATE jobs SET state='cancelled',cancel_requested=1,error='重新构建索引' "
            "WHERE kind='index' AND state IN ('paused','running')"
        )
    try:
        jobs.enqueue("index", space, settings)
    except JobConflict:
        pass
    with Worker(settings) as worker:
        worker.drain()

    retriever = Retriever(store, settings, Embedder(settings))
    retriever.prepare(space)
    print("index", retriever.status(space))

    manifest = {"space_id": space, "materials": []}
    with store.connect() as db:
        rows = db.execute(
            "SELECT m.id,m.name,m.format,m.revision,m.text,"
            "(SELECT count(*) FROM chunks c WHERE c.material_id=m.id),"
            "(SELECT count(*) FROM transcript_segments t WHERE t.material_id=m.id),"
            "(SELECT transcript_state FROM media md WHERE md.material_id=m.id) "
            "FROM materials m ORDER BY m.rowid"
        ).fetchall()
    for row in rows:
        entry = {
            "id": row[0],
            "name": row[1],
            "format": row[2],
            "revision": row[3],
            "characters": len(row[4]),
            "chunks": row[5],
            "segments": row[6],
            "transcript_state": row[7],
        }
        if entry["format"] == "video":
            media = store.media(space, entry["id"])
            entry["duration"] = round(media["duration"], 2)
            entry["transcript_source"] = media["transcript_source"]
        entry["source"] = store.material(space, entry["id"]).get("source")
        manifest["materials"].append(entry)
        print(
            f"  {entry['format']:6s} chunks={entry['chunks']:4d} segs={entry['segments']:5d} "
            f"chars={entry['characters']:6d} {entry['name']}"
        )
    out = root.parent / "corpus-manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("manifest", out)


if __name__ == "__main__":
    main()
