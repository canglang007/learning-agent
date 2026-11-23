import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from learning_agent.accounts import AccountStore
from learning_agent.drafts import DraftStore
from learning_agent.execution_store import ExecutionStore
from learning_agent.memory import MemoryStore
from learning_agent.run_store import (
    PENDING_SQL,
    SUCCESS,
    TERMINAL,
    RunStopped,
    RunStore,
    RuntimeConflict,
    now,
)


class Store(AccountStore, MemoryStore, ExecutionStore, RunStore, DraftStore):
    def __init__(self, root: Path):
        self.root = root
        (root / "materials").mkdir(parents=True, exist_ok=True)
        self.path = root / "workspace.db"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS spaces (id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS materials (
                    id TEXT PRIMARY KEY, space_id TEXT NOT NULL, name TEXT NOT NULL,
                    text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, space_id TEXT NOT NULL,
                    protocol TEXT NOT NULL DEFAULT '[]',
                    owner TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, run_id TEXT);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    state TEXT NOT NULL, events TEXT NOT NULL DEFAULT '[]', error TEXT);
                CREATE UNIQUE INDEX IF NOT EXISTS one_running_turn
                    ON runs(conversation_id) WHERE state='running';
                CREATE TABLE IF NOT EXISTS material_pages (
                    material_id TEXT NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,
                    width REAL NOT NULL, height REAL NOT NULL,
                    text_source TEXT NOT NULL DEFAULT 'text',
                    PRIMARY KEY(material_id, page));
                CREATE TABLE IF NOT EXISTS media (
                    material_id TEXT PRIMARY KEY, metadata TEXT NOT NULL,
                    state TEXT NOT NULL, error TEXT,
                    transcript_state TEXT NOT NULL DEFAULT 'missing',
                    transcript_source TEXT, transcript_error TEXT);
                CREATE TABLE IF NOT EXISTS transcript_segments (
                    material_id TEXT NOT NULL, seq INTEGER NOT NULL,
                    start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY(material_id,seq));
                CREATE TABLE IF NOT EXISTS transcript_revisions (
                    material_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    source TEXT NOT NULL, segments TEXT NOT NULL, alignment_notes TEXT NOT NULL,
                    PRIMARY KEY(material_id,revision));
                CREATE TABLE IF NOT EXISTS material_index (
                    material_id TEXT PRIMARY KEY, revision INTEGER, version INTEGER);
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, material_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_material ON chunks(material_id, ordinal);
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT NOT NULL, encoder TEXT NOT NULL, vector TEXT NOT NULL,
                    PRIMARY KEY(chunk_id,encoder));
                CREATE TABLE IF NOT EXISTS learning_feedback (
                    run_id TEXT PRIMARY KEY, target_run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, focus_text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS learning_notes (
                    seq INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, note TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS protocol_turns (
                    conversation_id TEXT NOT NULL, turn_seq INTEGER NOT NULL,
                    run_id TEXT UNIQUE, schema_version INTEGER NOT NULL,
                    profile TEXT NOT NULL, raw_input TEXT NOT NULL, groups_json TEXT NOT NULL,
                    replay_status TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(conversation_id,turn_seq));
                CREATE TABLE IF NOT EXISTS context_snapshots (
                    run_id TEXT NOT NULL, step INTEGER NOT NULL,
                    snapshot TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, PRIMARY KEY(run_id,step));
                CREATE TABLE IF NOT EXISTS learning_states (
                    conversation_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    state_json TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS context_summaries (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, covered_end INTEGER NOT NULL,
                    context_epoch INTEGER NOT NULL, state_revision INTEGER NOT NULL,
                    profile TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
                    result TEXT NOT NULL, created_at TEXT NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(materials)")}
            if "format" not in columns:
                db.execute("ALTER TABLE materials ADD COLUMN format TEXT NOT NULL DEFAULT 'text'")
            if "revision" not in columns:
                db.execute("ALTER TABLE materials ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
            if "source" not in columns:
                db.execute("ALTER TABLE materials ADD COLUMN source TEXT NOT NULL DEFAULT '{}'")
            media_columns = {row[1] for row in db.execute("PRAGMA table_info(media)")}
            if "transcript_revision" not in media_columns:
                db.execute(
                    "ALTER TABLE media ADD COLUMN transcript_revision INTEGER NOT NULL DEFAULT 0"
                )
            conversation_columns = {
                row[1] for row in db.execute("PRAGMA table_info(conversations)")
            }
            if "owner" not in conversation_columns:
                db.execute("ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
            page_columns = {row[1] for row in db.execute("PRAGMA table_info(material_pages)")}
            if "text_source" not in page_columns:
                db.execute(
                    "ALTER TABLE material_pages ADD COLUMN text_source TEXT NOT NULL DEFAULT 'text'"
                )
        self.init_memory()
        self.init_accounts()
        self.init_runtime()
        self.init_execution()
        self.init_drafts()

    @contextmanager
    def connect(self):
        from learning_agent.database import connection

        with connection(self.path) as db:
            yield db

    def space(self, name):
        """新建一个资料空间并返回它；不做同名去重。

        名字有歧义（读起来像"取空间"），但历史调用都按新建语义用。
        要查已有空间用 ``spaces()``，按名字挑。
        """
        item = {"id": str(uuid4()), "name": name}
        with self.connect() as db:
            db.execute("INSERT INTO spaces VALUES (:id, :name)", item)
        return item

    def spaces(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM spaces ORDER BY rowid DESC")]

    def require_space(self, space_id):
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM spaces WHERE id=?", (space_id,)).fetchone():
                raise LookupError("资料空间不存在")

    def add_material(self, space_id, name, text):
        self.require_space(space_id)
        item = {"id": str(uuid4()), "space_id": space_id, "name": name, "text": text}
        path = self.root / "materials" / (item["id"] + ".txt")
        path.write_text(text, encoding="utf-8")
        with self.connect() as db:
            db.execute(
                "INSERT INTO materials(id,space_id,name,text) VALUES (:id,:space_id,:name,:text)",
                item,
            )
        return {key: value for key, value in item.items() if key != "text"}

    def add_pdf(self, space_id, name, raw, min_text_characters=20):
        from learning_agent.pdf import extract_pages

        self.require_space(space_id)
        pages = extract_pages(raw)
        # A page without a usable text layer is a scan or a pure-image page. Record
        # that instead of storing an empty string that later looks like real content.
        for page in pages:
            page["text_source"] = (
                "text" if len(page["text"].strip()) >= min_text_characters else "none"
            )
        material_id = str(uuid4())
        text = "\n\n".join(f"[PDF 第{p['page']}页]\n{p['text']}" for p in pages)
        (self.root / "materials" / (material_id + ".pdf")).write_bytes(raw)
        with self.connect() as db:
            db.execute(
                "INSERT INTO materials(id,space_id,name,text,format) VALUES (?,?,?,?,'pdf')",
                (material_id, space_id, name, text),
            )
            db.executemany(
                "INSERT INTO material_pages VALUES (?,?,?,?,?,?)",
                [
                    (material_id, p["page"], p["text"], p["width"], p["height"], p["text_source"])
                    for p in pages
                ],
            )
        return {
            "id": material_id,
            "space_id": space_id,
            "name": name,
            "format": "pdf",
            "page_count": len(pages),
            "ocr_pending": sum(1 for p in pages if p["text_source"] == "none"),
        }

    def pages(self, space_id, material_id):
        self.material(space_id, material_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT page,length(text) AS text_characters,text_source FROM material_pages WHERE material_id=? ORDER BY page",
                    (material_id,),
                )
            ]

    def ocr_pending_pages(self, space_id, material_id):
        """Pages whose text was never attempted, in physical order.

        ``blank`` pages are excluded: recognition already ran there and found
        nothing, so retrying them would burn inference on every read of a
        figure-only page.
        """
        self.material(space_id, material_id)
        with self.connect() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT page FROM material_pages WHERE material_id=? AND text_source='none' ORDER BY page",
                    (material_id,),
                )
            ]

    def page_counts(self, space_id, material_id):
        """How many pages carry a text layer, OCR text, or nothing at all."""
        self.material(space_id, material_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT text_source,count(*) FROM material_pages WHERE material_id=? GROUP BY text_source",
                (material_id,),
            ).fetchall()
        counts = {"text": 0, "ocr": 0, "none": 0, "blank": 0}
        counts.update({row[0]: row[1] for row in rows})
        return counts

    def save_ocr_page(self, space_id, material_id, page, text):
        """Store OCR output for one page without touching the material revision.

        Two callers share this write. The background job saves every page and
        bumps the revision once in :meth:`finalize_ocr`, so an interrupted run
        resumes on the remaining pages instead of rebuilding the index after
        every single page. An on-demand read also lands here, and deliberately
        stops there: the page becomes readable without invalidating the index.
        """
        self.material(space_id, material_id)
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE material_pages SET text=?,text_source='ocr' WHERE material_id=? AND page=?",
                (text, material_id, page),
            )
            if not cursor.rowcount:
                raise LookupError("该资料没有此 PDF 页面")

    def mark_page_blank(self, space_id, material_id, page):
        """Record that recognition ran on a page and found no text at all.

        Keeping this distinct from ``none`` is what stops a figure-only page
        from being recognised again on every read.
        """
        self.material(space_id, material_id)
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE material_pages SET text_source='blank' WHERE material_id=? AND page=?",
                (material_id, page),
            )
            if not cursor.rowcount:
                raise LookupError("该资料没有此 PDF 页面")

    def finalize_ocr(self, space_id, material_id):
        """Rebuild the joined text and bump the revision so chunks are regenerated.

        Returns False when no page carries recognised text, in which case the
        material is left exactly as it was and no index work is queued.
        """
        self.material(space_id, material_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT page,text,text_source FROM material_pages WHERE material_id=? ORDER BY page",
                (material_id,),
            ).fetchall()
            if not any(row[2] == "ocr" for row in rows):
                return False
            text = "\n\n".join(f"[PDF 第{row[0]}页]\n{row[1]}" for row in rows)
            db.execute(
                "UPDATE materials SET text=?, revision=revision+1 WHERE id=?",
                (text, material_id),
            )
            return True

    def page(self, space_id, material_id, page):
        material = self.material(space_id, material_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM material_pages WHERE material_id=? AND page=?", (material_id, page)
            ).fetchone()
        if row is None:
            raise LookupError("该资料没有此 PDF 页面")
        return {**dict(row), "name": material["name"]}

    def page_image(self, space_id, material_id, page, region=None):
        from learning_agent.assets import atomic_png, image_target, png_size
        from learning_agent.pdf import render_page

        self.page(space_id, material_id, page)
        suffix = "full" if region is None else "-".join(format(v, ".6f") for v in region)
        asset = Path("materials") / material_id / f"page-{page}-{suffix}.png"
        output = image_target(self.root, asset)
        size = png_size(output)
        if size:
            width, height = size
        else:
            raw, width, height = render_page(
                self.root / "materials" / (material_id + ".pdf"), page, region
            )
            atomic_png(output, raw)
        return {
            "asset_path": str(asset),
            "mime_type": "image/png",
            "width": width,
            "height": height,
        }

    def materials(self, space_id):
        self.require_space(space_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT m.id,m.name,m.format,m.revision,(SELECT count(*) FROM material_pages p WHERE p.material_id=m.id) AS page_count,(SELECT count(*) FROM material_pages p WHERE p.material_id=m.id AND p.text_source='none') AS pages_without_text,media.state AS media_state,media.transcript_state FROM materials m LEFT JOIN media ON media.material_id=m.id WHERE m.space_id=? ORDER BY m.rowid",
                    (space_id,),
                )
            ]

    def material(self, space_id, material_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM materials WHERE space_id=? AND id=?", (space_id, material_id)
            ).fetchone()
        if row is None:
            raise LookupError("该空间内没有此资料")
        return dict(row)

    def add_media(self, space_id, name, raw, suffix):
        import tempfile

        with tempfile.NamedTemporaryFile(dir=self.root, suffix=suffix) as file:
            file.write(raw)
            file.flush()
            return self.add_media_file(space_id, name, Path(file.name), suffix, move=False)

    def add_media_file(
        self, space_id, name, path, suffix, material_id=None, max_seconds=14400, move=True
    ):
        from learning_agent.media import probe

        self.require_space(space_id)
        material_id = material_id or str(uuid4())
        with self.connect() as db:
            if db.execute(
                "SELECT 1 FROM materials WHERE id=? AND space_id=?", (material_id, space_id)
            ).fetchone():
                return self.media(space_id, material_id)
        directory = self.root / "materials" / material_id
        directory.mkdir(exist_ok=True)
        source = directory / ("original" + suffix)
        if path != source and path.exists():
            if move:
                path.replace(source)
            else:
                import shutil

                shutil.copyfile(path, source)
        metadata = probe(source, max_seconds=max_seconds)
        metadata["original_path"] = str(source.relative_to(self.root))
        metadata["preview_path"] = str((directory / "preview.mp4").relative_to(self.root))
        metadata["frame_index_seconds"] = (
            [
                round(
                    i * metadata["duration"] / max(1, min(60, int(metadata["duration"] / 30) + 1)),
                    3,
                )
                for i in range(max(1, min(60, int(metadata["duration"] / 30) + 1)))
            ]
            if metadata["has_video"]
            else []
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO materials(id,space_id,name,text,format) VALUES (?,?,?,'',?)",
                (material_id, space_id, name, "video" if metadata["has_video"] else "audio"),
            )
            db.execute(
                "INSERT INTO media(material_id,metadata,state) VALUES (?,?,'processing')",
                (material_id, json.dumps(metadata)),
            )
        return self.media(space_id, material_id)

    def media(self, space_id, material_id):
        material = self.material(space_id, material_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM media WHERE material_id=?", (material_id,)).fetchone()
            count = db.execute(
                "SELECT count(*) FROM transcript_segments WHERE material_id=?", (material_id,)
            ).fetchone()[0]
        if row is None:
            raise LookupError("此资料不是音视频")
        result = dict(row)
        metadata = json.loads(result.pop("metadata"))
        return {
            **result,
            **metadata,
            "name": material["name"],
            "format": material["format"],
            "segment_count": count,
            "playback_url": f"/api/spaces/{space_id}/materials/{material_id}/playback",
        }

    def request_transcode(self, space_id, material_id):
        """客户端说它解不了这份编码时，把媒体打回"待转码"。

        服务端只能按容器 + 编码判断"理论上能播"，但 AV1 这类编码在缺少硬件解码的
        浏览器（Safari + M1/M2）里确实放不出来。与其让用户对着黑屏猜，不如由客户端
        显式要一份 h264 预览。
        """
        self.media(space_id, material_id)
        preview = str((self.root / "materials" / material_id / "preview.mp4").relative_to(self.root))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT metadata FROM media WHERE material_id=?", (material_id,)
            ).fetchone()
            metadata = json.loads(row[0])
            metadata["force_transcode"] = True
            metadata["playback"] = "transcode"
            metadata["browser_playable"] = False
            metadata["preview_path"] = preview
            db.execute(
                "UPDATE media SET metadata=?,state='processing',error=NULL WHERE material_id=?",
                (json.dumps(metadata), material_id),
            )
        return self.media(space_id, material_id)

    def segments(self, space_id, material_id):
        self.media(space_id, material_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT start,end,text FROM transcript_segments WHERE material_id=? ORDER BY seq",
                    (material_id,),
                )
            ]

    def save_segments(
        self,
        space_id,
        material_id,
        segments,
        source,
        alignment_notes=None,
        job=None,
        allow_empty=False,
    ):
        from learning_agent.jobs import Jobs
        from learning_agent.media import valid_segments

        info = self.media(space_id, material_id)
        segments = valid_segments(segments, info["duration"], allow_empty=allow_empty)
        jobs = Jobs(self)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if job:
                jobs.check(job, db)
            else:
                # Uploaded captions supersede candidates, including an in-flight ASR response.
                db.execute(
                    "UPDATE jobs SET state='cancelled',cancel_requested=1,error='已由新字幕替代' WHERE material_id=? AND kind='transcribe' AND state IN ('queued','running','paused')",
                    (material_id,),
                )
            revision = (
                db.execute(
                    "SELECT transcript_revision FROM media WHERE material_id=?", (material_id,)
                ).fetchone()[0]
                + 1
            )
            db.execute("DELETE FROM transcript_segments WHERE material_id=?", (material_id,))
            self.invalidate_index(db, material_id)
            db.executemany(
                "INSERT INTO transcript_segments VALUES (?,?,?,?,?)",
                [(material_id, i, s["start"], s["end"], s["text"]) for i, s in enumerate(segments)],
            )
            db.execute(
                "UPDATE media SET transcript_state='ready',transcript_source=?,transcript_error=NULL,transcript_revision=? WHERE material_id=?",
                (source, revision, material_id),
            )
            if not segments:
                db.execute(
                    "UPDATE media SET transcript_state='silent' WHERE material_id=?", (material_id,)
                )
            db.execute(
                "INSERT INTO transcript_revisions VALUES (?,?,?,?,?)",
                (
                    material_id,
                    revision,
                    source,
                    json.dumps(segments, ensure_ascii=False),
                    json.dumps(alignment_notes or []),
                ),
            )

            metadata = json.loads(
                db.execute(
                    "SELECT metadata FROM media WHERE material_id=?", (material_id,)
                ).fetchone()[0]
            )
            metadata["alignment_notes"] = alignment_notes or []
            db.execute(
                "UPDATE media SET metadata=? WHERE material_id=?",
                (json.dumps(metadata), material_id),
            )
            if job:
                # The transcript revision changed above; finish against the checked input.
                db.execute(
                    "UPDATE jobs SET state='succeeded',stage='completed',completed=total,remote_pending=0,error=NULL,updated=? WHERE id=? AND token=?",
                    (time.time(), job["id"], job["token"]),
                )

    @staticmethod
    def invalidate_index(db, material_id):
        db.execute(
            "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE material_id=?)",
            (material_id,),
        )
        db.execute("DELETE FROM chunks WHERE material_id=?", (material_id,))
        db.execute("DELETE FROM material_index WHERE material_id=?", (material_id,))
        db.execute("UPDATE materials SET revision=revision+1 WHERE id=?", (material_id,))

    def set_source(self, space_id, material_id, source):
        self.material(space_id, material_id)
        with self.connect() as db:
            db.execute(
                "UPDATE materials SET source=? WHERE id=?", (json.dumps(source), material_id)
            )
            self.invalidate_index(db, material_id)
        return source

    def media_frame(self, space_id, material_id, seconds):
        from learning_agent.assets import image_target, png_size
        from learning_agent.media import MediaError, frame

        info = self.media(space_id, material_id)
        if not info["has_video"] or not 0 <= seconds < info["duration"]:
            raise MediaError("时间点超出视频范围或资料没有画面")
        asset = Path("materials") / material_id / f"frame-{seconds:.3f}.png"
        output = image_target(self.root, asset)
        if png_size(output) is None:
            frame(self.root / info["original_path"], output, seconds)
        width, height = png_size(output)
        return {
            "asset_path": str(asset),
            "mime_type": "image/png",
            "width": width,
            "height": height,
            "seconds": seconds,
        }

    def conversation(self, space_id, owner=""):
        """新建会话。

        ``owner`` 是归属键：登录用户为 ``u:<user_id>``，匿名模式是访客 cookie 值。
        空串表示所有人可见（示例会话、迁移前的历史、评测产物）。
        """
        self.require_space(space_id)
        item = {"id": str(uuid4()), "space_id": space_id, "owner": owner}
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations(id,space_id,owner) VALUES (:id,:space_id,:owner)", item
            )
        return {"id": item["id"], "space_id": space_id, "owner": owner}

    def get_conversation(self, conversation_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
        if row is None:
            raise LookupError("会话不存在")
        return {**dict(row), "protocol": json.loads(row["protocol"])}

    def conversations(self, space_id, owner=None):
        """列会话；给出 ``owner`` 时只返回他自己的和公共示例（``owner=''``）。"""
        self.require_space(space_id)
        sql = (
            "SELECT c.id,c.space_id,(SELECT substr(m.content,1,45) FROM messages m "
            "WHERE m.conversation_id=c.id AND m.role='user' ORDER BY m.seq LIMIT 1) AS title "
            "FROM conversations c WHERE c.space_id=?"
        )
        params = [space_id]
        if owner is not None:
            sql += " AND (c.owner='' OR c.owner=?)"
            params.append(owner)
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql + " ORDER BY c.rowid DESC", params)]

    def run_conversation(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT conversation_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise LookupError("问答任务不存在")
        return row["conversation_id"]

    def delete_conversation(self, conversation_id: str) -> None:
        """删除会话及其全部关联行（消息、轮次、事件、反馈等），不留孤儿行。

        只处理引用了 ``conversation_id`` 或该会话 ``run_id`` 的表，表名来自
        sqlite_master，不拼接外部输入。
        """
        self.get_conversation(conversation_id)
        with self.connect() as db:
            run_ids = [
                row["id"]
                for row in db.execute(
                    "SELECT id FROM runs WHERE conversation_id=?", (conversation_id,)
                )
            ]
            tables = [
                row["name"]
                for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            for table in tables:
                if table == "conversations" or table.startswith("sqlite_"):
                    continue
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if "conversation_id" in columns:
                    db.execute(f"DELETE FROM {table} WHERE conversation_id=?", (conversation_id,))
                if "run_id" in columns and run_ids:
                    placeholders = ",".join("?" * len(run_ids))
                    db.execute(f"DELETE FROM {table} WHERE run_id IN ({placeholders})", run_ids)
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def messages(self, conversation_id):
        self.get_conversation(conversation_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM messages WHERE conversation_id=? ORDER BY seq",
                    (conversation_id,),
                )
            ]

    def start_run(self, conversation_id, content, feedback=None):
        from learning_agent.learning import FeedbackError

        run_id = str(uuid4())
        with self.connect() as db:
            if feedback:
                target = db.execute(
                    "SELECT r.conversation_id,r.state FROM runs r JOIN messages m ON m.run_id=r.id AND m.role='assistant' WHERE r.id=?",
                    (feedback["target_run_id"],),
                ).fetchone()
                if (
                    not target
                    or target["conversation_id"] != conversation_id
                    or target["state"] not in {"completed", "budget_exhausted"}
                ):
                    raise FeedbackError("反馈需要指向当前会话中已完成的讲解")
            db.execute(
                "INSERT INTO runs(id,conversation_id,state) VALUES (?,?,'running')",
                (run_id, conversation_id),
            )
            db.execute(
                "INSERT INTO messages(conversation_id,role,content,run_id) VALUES (?,'user',?,?)",
                (conversation_id, content, run_id),
            )
            if feedback:
                db.execute(
                    "INSERT INTO learning_feedback VALUES (?,?,?,?)",
                    (run_id, feedback["target_run_id"], feedback["kind"], feedback["focus_text"]),
                )
        return run_id

    def save_events(self, run_id, events):
        with self.connect() as db:
            db.execute("UPDATE runs SET events=? WHERE id=?", (json.dumps(events), run_id))

    def finish_run(
        self,
        run_id,
        state,
        content,
        protocol,
        error=None,
        learning_notes=(),
        *,
        archive=None,
        state_update=None,
        expected_revision=None,
        owner_token=None,
        execution_guard=None,
    ):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row["state"] in TERMINAL:
                old = db.execute(
                    "SELECT content FROM messages WHERE run_id=? AND role='assistant'", (run_id,)
                ).fetchone()
                return {"state": row["state"], "answer": old[0] if old else ""}
            if owner_token is not None and row["owner_token"] != owner_token:
                raise RunStopped(interrupted=True)
            if row["cancel_requested"]:
                state, content, error = "cancelled", "本轮已停止，未提交讲解；可以重新提问。", None
                prior = db.execute(
                    "SELECT protocol FROM conversations WHERE id=?", (row["conversation_id"],)
                ).fetchone()
                question = db.execute(
                    "SELECT content FROM messages WHERE run_id=? AND role='user'", (run_id,)
                ).fetchone()
                protocol = json.loads(prior[0]) + [
                    {"role": "user", "content": question[0]},
                    {"role": "assistant", "content": content},
                ]
            if state in SUCCESS and expected_revision is not None:
                if execution_guard:
                    execution_guard(db)
                current = db.execute(
                    "SELECT revision FROM learning_states WHERE conversation_id=?",
                    (row["conversation_id"],),
                ).fetchone()
                if not current or current[0] != expected_revision:
                    raise RuntimeConflict("学习状态已改变，未提交本轮答案")
            db.execute(
                "UPDATE runs SET state=?,error=?,owner_token=NULL,phase=?,state_revision=state_revision+1,updated_at=? WHERE id=?",
                (state, error, state, now(), run_id),
            )
            db.execute(
                "UPDATE run_operations SET state='unknown',updated_at=? WHERE run_id=? AND state='sending'",
                (now(), run_id),
            )
            db.execute(
                "INSERT INTO messages(conversation_id,role,content,run_id,kind) VALUES (?,'assistant',?,?,?)",
                (
                    row["conversation_id"],
                    content,
                    run_id,
                    "answer" if state in SUCCESS else "system",
                ),
            )
            db.execute(
                "UPDATE conversations SET protocol=? WHERE id=?",
                (json.dumps(protocol), row["conversation_id"]),
            )
            if state in SUCCESS:
                db.execute(
                    "UPDATE run_drafts SET status='committed',updated_at=? WHERE run_id=? AND status IN ('generating','validating')",
                    (now(), run_id),
                )
            else:
                self.revoke_draft_in(db, run_id, state)
            if archive is not None:
                seq = db.execute(
                    "SELECT COALESCE(MAX(turn_seq),0)+1 FROM protocol_turns WHERE conversation_id=?",
                    (row["conversation_id"],),
                ).fetchone()[0]
                db.execute(
                    "INSERT INTO protocol_turns VALUES (?,?,?,1,?,?,?,?,?)",
                    (
                        row["conversation_id"],
                        seq,
                        run_id,
                        json.dumps(archive["profile"]),
                        json.dumps({**archive["input"], "answer": content}),
                        json.dumps(archive["groups"]),
                        "ready" if state in SUCCESS else "failed",
                        datetime.now(UTC).isoformat(),
                    ),
                )
            if state in {"completed", "budget_exhausted"}:
                db.executemany(
                    "INSERT INTO learning_notes(conversation_id,run_id,note) VALUES (?,?,?)",
                    [
                        (row["conversation_id"], run_id, json.dumps(note, ensure_ascii=False))
                        for note in learning_notes
                    ],
                )
                if state_update is not None:
                    changed = db.execute(
                        "UPDATE learning_states SET revision=?,state_json=?,updated_at=? "
                        "WHERE conversation_id=? AND revision=?",
                        (
                            state_update["revision"],
                            json.dumps(state_update),
                            datetime.now(UTC).isoformat(),
                            row["conversation_id"],
                            expected_revision,
                        ),
                    ).rowcount
                    if changed != 1:
                        raise RuntimeConflict("学习状态已改变，未提交本轮答案")
            self.event_in(
                db,
                run_id,
                "run.completed" if state in SUCCESS else "run." + state,
                {"state": state},
            )
        return {"state": state, "answer": content}

    def learning_state(self, conversation_id):
        self.get_conversation(conversation_id)
        with self.connect() as db:
            notes = db.execute(
                "SELECT run_id,note FROM learning_notes WHERE conversation_id=? ORDER BY seq DESC LIMIT 6",
                (conversation_id,),
            ).fetchall()
            feedback = db.execute(
                "SELECT f.*,m.content,r.state AS response_state FROM learning_feedback f JOIN runs r ON r.id=f.run_id JOIN messages m ON m.run_id=f.run_id AND m.role='user' WHERE r.conversation_id=? ORDER BY m.seq DESC LIMIT 8",
                (conversation_id,),
            ).fetchall()
        attempts = [{"run_id": row["run_id"], **json.loads(row["note"])} for row in reversed(notes)]
        cutoff = self.context_settings(conversation_id)["history_after"]
        if cutoff >= 0:
            with self.connect() as db:
                allowed = {
                    r[0]
                    for r in db.execute(
                        "SELECT run_id FROM protocol_turns WHERE conversation_id=? AND turn_seq>?",
                        (conversation_id, cutoff),
                    )
                }
            attempts = [a for a in attempts if a["run_id"] in allowed]
            feedback = [f for f in feedback if f["run_id"] in allowed]
        state = self.current_learning_state(conversation_id)
        return {
            "working_note": attempts[-1] if attempts and not state["reset"] else None,
            "attempts": attempts,
            "feedback": [dict(row) for row in reversed(feedback)],
            "state": state,
            "revision": state["revision"],
            "detail": "工作记录是Agent可修正的判断；明白了是用户对所指回答的自述，不代表客观掌握。",
        }

    def current_learning_state(self, conversation_id):
        from learning_agent.learning import apply_learning_update, empty_state

        self.get_conversation(conversation_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state_json FROM learning_states WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            if row:
                state = json.loads(row[0])
                state.setdefault(
                    "goal_pinned",
                    bool(state["goal"] and state["goal"]["origin"] == "user_explicit"),
                )
                for ref in state["evidence_refs"]:
                    current = db.execute(
                        "SELECT revision FROM materials WHERE id=? AND space_id="
                        "(SELECT space_id FROM conversations WHERE id=?)",
                        (ref["material_id"], conversation_id),
                    ).fetchone()
                    ref["version_status"] = (
                        "current_version"
                        if current and current[0] == ref["material_revision"]
                        else "stale_or_missing"
                    )
                return state
            state = empty_state()
            note = db.execute(
                "SELECT run_id,note FROM learning_notes WHERE conversation_id=? ORDER BY seq DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if note:
                data = json.loads(note["note"])
                state = apply_learning_update(
                    state,
                    {"active_topic": data["topic"], "confusion": data["confusion"]},
                    run_id=note["run_id"],
                    note=data,
                )
            db.execute(
                "INSERT INTO learning_states VALUES (?,?,?,?)",
                (
                    conversation_id,
                    state["revision"],
                    json.dumps(state),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return state

    def edit_learning_state(self, conversation_id, edit):
        from learning_agent.learning import LearningConflict, apply_learning_update, empty_state

        self.current_learning_state(conversation_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                f"SELECT 1 FROM runs WHERE conversation_id=? AND state IN {PENDING_SQL}",
                (conversation_id,),
            ).fetchone():
                raise LearningConflict("正在回答，请完成后再修改学习状态")
            row = db.execute(
                "SELECT revision,state_json FROM learning_states WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row["revision"] != edit["revision"]:
                raise LearningConflict("学习状态已更新，请刷新后再修改")
            if edit.get("reset"):
                prior = json.loads(row["state_json"])
                state = empty_state()
                state.update(
                    revision=prior["revision"] + 1,
                    context_epoch=prior["context_epoch"] + 1,
                    reset=True,
                )
            else:
                state = apply_learning_update(json.loads(row["state_json"]), edit, user=True)
            db.execute(
                "UPDATE learning_states SET revision=?,state_json=?,updated_at=? WHERE conversation_id=?",
                (
                    state["revision"],
                    json.dumps(state),
                    datetime.now(UTC).isoformat(),
                    conversation_id,
                ),
            )
            db.execute(
                "UPDATE context_summaries SET status='invalidated' WHERE conversation_id=? AND status='ready'",
                (conversation_id,),
            )
        return self.learning_state(conversation_id)

    def recall_learning_context(
        self,
        conversation_id,
        run_id,
        role,
        start_char,
        max_chars,
        turn_seq=None,
        memory_sources=(),
    ):
        self.get_conversation(conversation_id)
        with self.connect() as db:
            if (run_id is None) == (turn_seq is None):
                raise ValueError("run_id与turn_seq需且只能提供一个")
            if run_id is None:
                source = db.execute(
                    "SELECT run_id FROM protocol_turns WHERE conversation_id=? AND turn_seq=?",
                    (conversation_id, turn_seq),
                ).fetchone()
                run_id = source[0] if source else None
            source_conversation = conversation_id
            if run_id in memory_sources:
                source = db.execute(
                    "SELECT conversation_id FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if source:
                    source_conversation = source[0]
            row = db.execute(
                "SELECT m.content,r.state FROM messages m JOIN runs r ON r.id=m.run_id "
                f"WHERE m.conversation_id=? AND m.run_id=? AND m.role=? AND r.state NOT IN {PENDING_SQL}",
                (source_conversation, run_id, role),
            ).fetchone()
        if row is None:
            raise LookupError("当前会话没有此已完成回合的可见原话")
        text = row["content"]
        if start_char > len(text):
            raise ValueError("原话读取偏移超出长度")
        return {
            "run_id": run_id,
            "conversation_id": source_conversation,
            "role": role,
            "response_state": row["state"],
            "content": text[start_char : start_char + max_chars],
            "start_char": start_char,
            "end_char": min(len(text), start_char + max_chars),
            "total_chars": len(text),
            "truncated": start_char > 0 or start_char + max_chars < len(text),
            "historical_only": True,
            "detail": "历史可见原话，不是资料核实或当前视觉观察，也不重新启用旧偏好。当前问题、明确状态和有效主题记忆优先；省略部分可按偏移继续读取。",
        }

    def feedback_target(self, conversation_id, target_run_id):
        from learning_agent.learning import FeedbackError

        with self.connect() as db:
            row = db.execute(
                "SELECT m.content FROM messages m JOIN runs r ON r.id=m.run_id WHERE r.id=? AND r.conversation_id=? AND m.role='assistant'",
                (target_run_id, conversation_id),
            ).fetchone()
        if not row:
            raise FeedbackError("反馈所指的讲解不在当前会话")
        return {
            "run_id": target_run_id,
            "answer": row["content"],
            "truncated": False,
        }

    def context_turns(self, conversation_id):
        """Idempotently retain the unsplittable legacy protocol; don't infer turn roles."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            conv = db.execute(
                "SELECT protocol FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            if conv is None:
                raise LookupError("会话不存在")
            if not db.execute(
                "SELECT 1 FROM protocol_turns WHERE conversation_id=? AND turn_seq=0",
                (conversation_id,),
            ).fetchone():
                visible = [
                    dict(m)
                    for m in db.execute(
                        "SELECT m.role,m.content FROM messages m LEFT JOIN runs r ON r.id=m.run_id "
                        "WHERE m.conversation_id=? AND m.role IN ('user','assistant') "
                        f"AND (m.run_id IS NULL OR r.state NOT IN {PENDING_SQL}) ORDER BY m.seq",
                        (conversation_id,),
                    )
                ]
                db.execute(
                    "INSERT INTO protocol_turns VALUES (?,0,NULL,1,'{}',?,?,'legacy',?)",
                    (
                        conversation_id,
                        json.dumps({"visible": visible}),
                        json.dumps(
                            [{"id": "legacy", "kind": "legacy", "messages": json.loads(conv[0])}]
                        ),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            rows = db.execute(
                "SELECT * FROM protocol_turns WHERE conversation_id=? ORDER BY turn_seq",
                (conversation_id,),
            ).fetchall()
        turns = []
        for row in rows:
            raw = json.loads(row["raw_input"])
            groups = json.loads(row["groups_json"])
            if row["turn_seq"] == 0:
                if not raw["visible"]:
                    continue
                visible = raw["visible"]
            else:
                visible = [
                    {"role": "user", "content": raw["question"]},
                    {
                        "role": "assistant",
                        "content": raw["answer"]
                        if row["replay_status"] == "ready"
                        else "[上一轮执行失败，未完成讲解：" + raw["answer"] + "]",
                    },
                ]
            turns.append(
                {
                    "turn_seq": row["turn_seq"],
                    "run_id": row["run_id"],
                    "raw_input": raw,
                    "profile": json.loads(row["profile"]),
                    "groups": groups,
                    "replay_status": row["replay_status"],
                    "visible": visible,
                    "delivered_answer": raw.get("answer")
                    if groups and groups[-1]["kind"] == "tools" and row["replay_status"] == "ready"
                    else None,
                }
            )
        return turns

    def save_context(self, run_id, step, snapshot):
        with self.connect() as db:
            db.execute(
                "INSERT INTO context_snapshots VALUES (?,?,?,'{}',?) ON CONFLICT(run_id,step) DO UPDATE SET snapshot=excluded.snapshot",
                (run_id, step, json.dumps(snapshot), datetime.now(UTC).isoformat()),
            )

    def save_summary(self, record, *, run_id=None, owner_token=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if run_id is not None:
                run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
                if (
                    not run
                    or run["owner_token"] != owner_token
                    or run["state"] != "running"
                    or run["cancel_requested"]
                ):
                    record["status"] = "interrupted"
            db.execute(
                "INSERT INTO context_summaries VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload=excluded.payload,result=excluded.result",
                (
                    record["id"],
                    record["conversation_id"],
                    record["covered_end"],
                    record["context_epoch"],
                    record["state_revision"],
                    json.dumps(record["profile"]),
                    record["status"],
                    json.dumps(record["payload"]),
                    json.dumps(record["result"]),
                    record["created_at"],
                ),
            )

    def latest_summary_attempt(self, conversation_id, epoch, profile, *, ready=False):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM context_summaries WHERE conversation_id=? AND context_epoch=? "
                + ("AND status='ready' " if ready else "")
                + "ORDER BY rowid DESC",
                (conversation_id, epoch),
            ).fetchall()
        for row in rows:
            if json.loads(row["profile"]) == profile:
                return {
                    **dict(row),
                    "profile": json.loads(row["profile"]),
                    "payload": json.loads(row["payload"]),
                    "result": json.loads(row["result"]),
                }
        return None

    def context_result(self, run_id, step, result):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT result FROM context_snapshots WHERE run_id=? AND step=?", (run_id, step)
            ).fetchone()
            merged = {**(json.loads(prior[0]) if prior else {}), **result}
            db.execute(
                "UPDATE context_snapshots SET result=? WHERE run_id=? AND step=?",
                (json.dumps(merged), run_id, step),
            )

    def run_context(self, run_id):
        self.run(run_id)
        with self.connect() as db:
            return [
                {
                    "step": r["step"],
                    "created_at": r["created_at"],
                    **json.loads(r["snapshot"]),
                    "result": json.loads(r["result"]),
                }
                for r in db.execute(
                    "SELECT * FROM context_snapshots WHERE run_id=? ORDER BY step", (run_id,)
                )
            ]

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise LookupError("运行不存在")
        return {**dict(row), "events": json.loads(row["events"])}

    def reference_context(self, run_id):
        """Only this conversation's observations up through the requested turn."""
        with self.connect() as db:
            run = db.execute(
                "SELECT rowid,conversation_id FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise LookupError("运行不存在")
            answer = db.execute(
                "SELECT content FROM messages WHERE run_id=? AND role='assistant'", (run_id,)
            ).fetchone()
            rows = db.execute(
                "SELECT events FROM runs WHERE conversation_id=? AND rowid<=? ORDER BY rowid",
                (run["conversation_id"], run["rowid"]),
            ).fetchall()
        return (answer[0] if answer else ""), [
            event for row in rows for event in json.loads(row[0])
        ]
