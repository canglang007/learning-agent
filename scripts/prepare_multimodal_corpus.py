"""把无声视频加入评测语料并重建索引（一次性准备脚本）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from learning_agent.config import Settings
from learning_agent.retrieval import Embedder, Retriever
from learning_agent.store import Store

root = Path("data/evaluations/multimodal-v1/corpus")
source = Path("data/evaluations/multimodal-v1/assets/cs336-prenorm-silent.mp4")

settings = Settings(
    _env_file=None,
    data_root=root,
    retrieval_mode="hybrid",
    embedding_provider="local",
)
store = Store(root)
space = store.spaces()[0]["id"]
print("space", space)

existing = {m["name"] for m in store.materials(space)}
if source.name in existing:
    print("已存在，跳过添加")
else:
    item = store.add_media_file(space, source.name, source, ".mp4", move=False)
    print("added", item["material_id"], "state", item["state"], "duration", item["duration"])

    from learning_agent.jobs import Jobs
    from learning_agent.worker import Worker

    Jobs(store).enqueue("preview", space, settings, item["material_id"])
    with Worker(settings) as worker:
        worker.drain()
    media = store.media(space, item["material_id"])
    print("after worker state", media["state"], "segments", media["segment_count"])
    with store.connect() as db:
        db.execute(
            "UPDATE materials SET source=? WHERE id=?",
            (json.dumps({"url": None, "start_seconds": 0}), item["material_id"]),
        )

retriever = Retriever(store, settings, Embedder(settings))
retriever.prepare(space)
print("index", retriever.status(space))

with store.connect() as db:
    for mid, name, fmt in db.execute("select id,name,format from materials"):
        chunks = db.execute("select count(*) from chunks where material_id=?", (mid,)).fetchone()[0]
        print(f"  {fmt:6s} chunks={chunks:4d} {name}")
