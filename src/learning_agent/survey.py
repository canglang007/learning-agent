"""试用问卷：一个页面，答案进同一个 SQLite，可导出成引用材料。

不接第三方表单，是因为这里的问题和这个 Agent 绑得很紧（哪一步没懂、
有没有觉得它在编），放在同一个库里才能和 run 对照着看。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS survey_responses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    role TEXT,
    rating INTEGER,
    stuck TEXT,
    fabricated TEXT,
    fabricating_detail TEXT,
    reuse TEXT,
    contact TEXT
)
"""

PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learning Agent · 试用反馈</title>
<link rel="icon" href="data:,">
<style>
:root{font:15px/1.7 system-ui;color:#20312d;background:#f5f6f2}
body{margin:0;padding:28px}
main{max-width:640px;margin:auto;background:white;border:1px solid #dce3dc;border-radius:12px;padding:26px}
h1{font-size:20px;margin:0 0 6px}p{color:#53675e;margin:6px 0}
label{display:block;margin-top:20px;font-weight:600}
input,select,textarea{font:inherit;width:100%;padding:9px;border:1px solid #cbd5cc;border-radius:6px;margin-top:6px;box-sizing:border-box}
textarea{resize:vertical}
button{font:inherit;margin-top:24px;padding:10px 22px;border:1px solid #cbd5cc;border-radius:6px;background:#e7efe7;cursor:pointer}
small{display:block;color:#64756b;margin-top:16px}
#done{display:none;margin-top:18px;padding:12px;background:#eef4ee;border-radius:8px}
a{color:#38634b}
</style>
<main>
<h1>试用反馈</h1>
<p>用过之后花一分钟填一下。没有标准答案，觉得不对的地方直说就行。</p>
<form id="form">
<label>你的背景
<select name="role">
<option value="">不填</option>
<option>本科生</option>
<option>研究生</option>
<option>已工作</option>
<option>其他</option>
</select>
</label>

<label>刚才的讲解，帮你把问题弄懂了吗？<span id="rating-value"></span>
<input name="rating" type="range" min="1" max="5" step="1" value="3">
</label>

<label>哪一步最卡、最想让它讲清楚的地方？
<textarea name="stuck" rows="3" maxlength="1000" placeholder="比如：它讲公式那一段跳得太快"></textarea>
</label>

<label>有没有哪句话让你觉得它在编？
<select name="fabricated">
<option value="">没注意</option>
<option>有</option>
<option>没有</option>
</select>
</label>

<label>如果选了「有」，是哪里？
<textarea name="fabricating_detail" rows="2" maxlength="1000"></textarea>
</label>

<label>以后还会用它学东西吗？
<select name="reuse">
<option value="">不填</option>
<option>会</option>
<option>看情况</option>
<option>不会</option>
</select>
</label>

<label>联系方式（选填，想追问细节时用）
<input name="contact" maxlength="120">
</label>

<button type="submit">提交</button>
<small>答案存在这台机器上，只用于改进这个项目，不会发给第三方。</small>
</form>
<div id="done">已收到，谢谢。</div>
<p><a href="/">回到对话框</a></p>
</main>
<script>
const form = document.getElementById('form');
const rating = form.elements.rating;
const out = document.getElementById('rating-value');
const paint = () => out.textContent = '（' + rating.value + '/5）';
paint(); rating.addEventListener('input', paint);
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  const response = await fetch('/api/survey', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  if (!response.ok) { alert('提交失败：' + (await response.text())); return; }
  form.hidden = true;
  document.getElementById('done').style.display = 'block';
});
</script>
</html>
"""


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def save(store, payload):
    record = {
        "id": str(uuid4()),
        "created_at": _now(),
        "role": (payload.get("role") or None),
        "rating": payload.get("rating"),
        "stuck": (payload.get("stuck") or "").strip()[:1000] or None,
        "fabricated": (payload.get("fabricated") or None),
        "fabricating_detail": (payload.get("fabricating_detail") or "").strip()[:1000] or None,
        "reuse": (payload.get("reuse") or None),
        "contact": (payload.get("contact") or "").strip()[:120] or None,
    }
    rating = record["rating"]
    if rating is not None:
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            raise ValueError("评分需要是 1 到 5 的整数") from None
        if not 1 <= rating <= 5:
            raise ValueError("评分需要是 1 到 5 的整数")
        record["rating"] = rating
    with store.connect() as db:
        db.execute(SCHEMA)
        db.execute(
            "INSERT INTO survey_responses VALUES (:id, :created_at, :role, :rating, :stuck,"
            " :fabricated, :fabricating_detail, :reuse, :contact)",
            record,
        )
    return record


def summary(store):
    with store.connect() as db:
        db.execute(SCHEMA)
        rows = [dict(item) for item in db.execute("SELECT * FROM survey_responses")]
    ratings = sorted(row["rating"] for row in rows if row["rating"] is not None)
    return {
        "responses": len(rows),
        "rated": len(ratings),
        "rating_mean": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "fabricated_reports": sum(1 for row in rows if row["fabricated"] == "有"),
        "reuse_yes": sum(1 for row in rows if row["reuse"] == "会"),
        "reuse_maybe": sum(1 for row in rows if row["reuse"] == "看情况"),
    }


def export(store, path):
    with store.connect() as db:
        db.execute(SCHEMA)
        rows = [
            dict(item) for item in db.execute("SELECT * FROM survey_responses ORDER BY created_at")
        ]
    payload = {"generated_at": _now(), "summary": summary(store), "responses": rows}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return payload


def install(app, store):
    @app.get("/survey", response_class=HTMLResponse)
    async def page():
        return PAGE

    @app.post("/api/survey", status_code=201)
    async def submit(request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"detail": "需要JSON"}, status_code=422)
        try:
            return save(store, payload if isinstance(payload, dict) else {})
        except ValueError as error:
            return JSONResponse({"detail": str(error)}, status_code=422)

    @app.get("/api/survey")
    async def stats():
        return summary(store)


def main():
    import argparse

    from learning_agent.store import Store

    parser = argparse.ArgumentParser(description="导出试用问卷结果")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export(Store(args.workspace), args.output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
