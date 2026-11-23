"""校验评测集里的标注目标是否真的可达。

两类标注各有各的失效方式，都不报错、只让分数偏低：

- PDF 页不存在（页码超范围、或者这一页本来就是纯图页，chunks=0）；
- 视频区间跨了两个分块。评测判定的规则是「某个分块完整包含目标区间」，
  跨块区间永远不成立，而 ``any()`` 断言语义会让它看起来只是"没检索到"。

v3 就吃过一次亏：``runtime`` 标注了一个在任何时间基准下都到不了的时间点，
被 ``any()`` 掩盖成漏检。换成整讲原片后时间基重排，同类错误更容易复发，
所以把校验写成脚本，改标注后跑一次。

用法：
    python scripts/check_annotation_targets.py --queries evals/retrieval-v4.json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_assets import DEFAULT_ASSETS


def load(corpus):
    db = sqlite3.connect(Path(corpus) / "workspace.db")
    db.row_factory = sqlite3.Row
    materials = {row["name"]: dict(row) for row in db.execute("SELECT * FROM materials")}
    chunks, spans, pages = {}, {}, {}
    for row in db.execute("SELECT material_id, payload FROM chunks ORDER BY ordinal"):
        item = json.loads(row["payload"])
        chunks[row["material_id"]] = chunks.get(row["material_id"], 0) + 1
        if "start_seconds" in item:
            spans.setdefault(row["material_id"], []).append(
                (item["start_seconds"], item["end_seconds"])
            )
    for row in db.execute("SELECT material_id, page FROM material_pages"):
        pages.setdefault(row["material_id"], set()).add(row["page"])
    return materials, chunks, spans, pages


def main():
    parser = argparse.ArgumentParser(description="校验标注目标可达性")
    parser.add_argument("--queries", default="evals/retrieval-v4.json")
    parser.add_argument("--corpus", default="data/evaluations/multimodal-v1/corpus")
    args = parser.parse_args()

    materials, chunk_count, spans, pages = load(args.corpus)
    by_name = DEFAULT_ASSETS
    missing = sorted({k for k in by_name if by_name[k] not in materials})
    for key in missing:
        print(f"资产缺失 {key} -> {by_name[key]}")

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    failures, notes = [], []
    for query in queries:
        expected = query.get("expected") or []
        if not expected:
            continue
        if query.get("unretrievable"):
            notes.append(f"{query['id']}: 已声明 unretrievable，不计入召回分母")
            continue
        for target in expected:
            name = by_name.get(target["asset"])
            material = materials.get(name)
            if material is None:
                failures.append(f"{query['id']}: 资产 {target['asset']}->{name} 不在语料")
                continue
            if target.get("pages"):
                have = pages.get(material["id"], set())
                gone = [p for p in target["pages"] if p not in have]
                if gone:
                    failures.append(f"{query['id']}: {name} 没有第 {gone} 页")
                if not chunk_count.get(material["id"]):
                    notes.append(
                        f"{query['id']}: {name} 无分块（纯图页），未标 unretrievable 时会必然漏检"
                    )
            if target.get("seconds"):
                a, b = target["seconds"]
                intervals = spans.get(material["id"], [])
                if not any(x <= a and b <= y for x, y in intervals):
                    near = sorted(intervals, key=lambda s: abs(s[0] - a))[:2]
                    failures.append(
                        f"{query['id']}: {name} 区间 [{a}, {b}] 不在单个分块内；"
                        f"邻近分块 {[tuple(round(v, 1) for v in s) for s in near]}"
                    )
    for line in notes:
        print("提示", line)
    for line in failures:
        print("失败", line)
    print(f"\n{len(queries)} 题，{len(failures)} 处不可达")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
