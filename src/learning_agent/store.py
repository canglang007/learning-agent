import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Store:
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
                    id TEXT PRIMARY KEY, space_id TEXT NOT NULL, protocol TEXT NOT NULL DEFAULT '[]');
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
            """)
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

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA busy_timeout=5000")
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def space(self, name):
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

    def add_pdf(self, space_id, name, raw):
        from learning_agent.pdf import extract_pages

        self.require_space(space_id)
        pages = extract_pages(raw)
        material_id = str(uuid4())
        text = "\n\n".join(f"[PDF 第{p['page']}页]\n{p['text']}" for p in pages)
        (self.root / "materials" / (material_id + ".pdf")).write_bytes(raw)
        with self.connect() as db:
            db.execute(
                "INSERT INTO materials(id,space_id,name,text,format) VALUES (?,?,?,?,'pdf')",
                (material_id, space_id, name, text),
            )
            db.executemany(
                "INSERT INTO material_pages VALUES (?,?,?,?,?)",
                [(material_id, p["page"], p["text"], p["width"], p["height"]) for p in pages],
            )
        return {
            "id": material_id,
            "space_id": space_id,
            "name": name,
            "format": "pdf",
            "page_count": len(pages),
        }

    def pages(self, space_id, material_id):
        self.material(space_id, material_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT page,length(text) AS text_characters FROM material_pages WHERE material_id=? ORDER BY page",
                    (material_id,),
                )
            ]

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
        from learning_agent.pdf import render_page

        self.page(space_id, material_id, page)
        suffix = "full" if region is None else "-".join(format(v, ".6f") for v in region)
        asset = Path("materials") / material_id / f"page-{page}-{suffix}.png"
        output = self.root / asset
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            from PIL import Image

            with Image.open(output) as image:
                width, height = image.size
        else:
            raw, width, height = render_page(
                self.root / "materials" / (material_id + ".pdf"), page, region
            )
            output.write_bytes(raw)
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
                    "SELECT m.id,m.name,m.format,(SELECT count(*) FROM material_pages p WHERE p.material_id=m.id) AS page_count,media.state AS media_state,media.transcript_state FROM materials m LEFT JOIN media ON media.material_id=m.id WHERE m.space_id=? ORDER BY m.rowid",
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
        from learning_agent.media import probe

        self.require_space(space_id)
        material_id = str(uuid4())
        directory = self.root / "materials" / material_id
        directory.mkdir()
        source = directory / ("original" + suffix)
        source.write_bytes(raw)
        try:
            metadata = probe(source)
        except Exception:
            source.unlink(missing_ok=True)
            directory.rmdir()
            raise
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
        self, space_id, material_id, segments, source, alignment_notes=None, job=None
    ):
        from learning_agent.jobs import Jobs
        from learning_agent.media import valid_segments

        info = self.media(space_id, material_id)
        segments = valid_segments(segments, info["duration"])
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
        from PIL import Image

        from learning_agent.media import MediaError, frame

        info = self.media(space_id, material_id)
        if not info["has_video"] or not 0 <= seconds < info["duration"]:
            raise MediaError("时间点超出视频范围或资料没有画面")
        asset = Path("materials") / material_id / f"frame-{seconds:.3f}.png"
        output = self.root / asset
        if not output.exists():
            frame(self.root / info["original_path"], output, seconds)
        with Image.open(output) as image:
            width, height = image.size
        return {
            "asset_path": str(asset),
            "mime_type": "image/png",
            "width": width,
            "height": height,
            "seconds": seconds,
        }

    def conversation(self, space_id):
        self.require_space(space_id)
        item = {"id": str(uuid4()), "space_id": space_id}
        with self.connect() as db:
            db.execute("INSERT INTO conversations(id,space_id) VALUES (:id,:space_id)", item)
        return item

    def get_conversation(self, conversation_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
        if row is None:
            raise LookupError("会话不存在")
        return {**dict(row), "protocol": json.loads(row["protocol"])}

    def conversations(self, space_id):
        self.require_space(space_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT c.id,c.space_id,(SELECT substr(m.content,1,45) FROM messages m WHERE m.conversation_id=c.id AND m.role='user' ORDER BY m.seq LIMIT 1) AS title FROM conversations c WHERE c.space_id=? ORDER BY c.rowid DESC",
                    (space_id,),
                )
            ]

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

    def finish_run(self, run_id, state, content, protocol, error=None, learning_notes=()):
        with self.connect() as db:
            row = db.execute("SELECT conversation_id FROM runs WHERE id=?", (run_id,)).fetchone()
            db.execute("UPDATE runs SET state=?,error=? WHERE id=?", (state, error, run_id))
            db.execute(
                "INSERT INTO messages(conversation_id,role,content,run_id) VALUES (?,'assistant',?,?)",
                (row["conversation_id"], content, run_id),
            )
            db.execute(
                "UPDATE conversations SET protocol=? WHERE id=?",
                (json.dumps(protocol), row["conversation_id"]),
            )
            if state in {"completed", "budget_exhausted"}:
                db.executemany(
                    "INSERT INTO learning_notes(conversation_id,run_id,note) VALUES (?,?,?)",
                    [
                        (row["conversation_id"], run_id, json.dumps(note, ensure_ascii=False))
                        for note in learning_notes
                    ],
                )

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
        return {
            "working_note": attempts[-1] if attempts else None,
            "attempts": attempts,
            "feedback": [dict(row) for row in reversed(feedback)],
            "detail": "工作记录是Agent可修正的判断；明白了是用户对所指回答的自述，不代表客观掌握。",
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
            "answer": row["content"][:8000],
            "truncated": len(row["content"]) > 8000,
        }

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
