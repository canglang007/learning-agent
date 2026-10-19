import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, root: Path):
        self.root = root
        (root / "materials").mkdir(parents=True, exist_ok=True)
        self.path = root / "workspace.db"
        with self.connect() as db:
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
                CREATE TABLE IF NOT EXISTS material_index (
                    material_id TEXT PRIMARY KEY, revision INTEGER, version INTEGER);
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, material_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_material ON chunks(material_id, ordinal);
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT NOT NULL, encoder TEXT NOT NULL, vector TEXT NOT NULL,
                    PRIMARY KEY(chunk_id,encoder));
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(materials)")}
            if "format" not in columns:
                db.execute("ALTER TABLE materials ADD COLUMN format TEXT NOT NULL DEFAULT 'text'")
            if "revision" not in columns:
                db.execute("ALTER TABLE materials ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
            if "source" not in columns:
                db.execute("ALTER TABLE materials ADD COLUMN source TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
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

    def recover_media(self):
        # Local background work is not a persistent queue. Recover visibly, offer retry.
        with self.connect() as db:
            db.execute(
                "UPDATE media SET state='failed',error='服务中断，请重试准备' WHERE state='processing'"
            )
            db.execute(
                "UPDATE media SET transcript_state='failed',transcript_error='服务中断，请重试转写' WHERE transcript_state='running'"
            )

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

    def prepare_media(self, space_id, material_id):
        from learning_agent.media import MediaError, prepare

        info = self.media(space_id, material_id)
        try:
            prepare(self.root / info["original_path"], self.root / info["preview_path"], info)
            state, error = "ready", None
        except MediaError as exception:
            state, error = "failed", str(exception)
        with self.connect() as db:
            db.execute(
                "UPDATE media SET state=?,error=? WHERE material_id=?", (state, error, material_id)
            )

    def claim_media_job(self, space_id, material_id, transcription=False):
        from learning_agent.media import MediaError

        info = self.media(space_id, material_id)
        if transcription and not info["has_audio"]:
            raise MediaError("该视频没有音轨，可查看画面或上传已有字幕")
        column = "transcript_state" if transcription else "state"
        running = "running" if transcription else "processing"
        with self.connect() as db:
            changed = db.execute(
                f"UPDATE media SET {column}=? WHERE material_id=? AND {column}!=?",
                (running, material_id, running),
            ).rowcount
        return bool(changed)

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
        self, space_id, material_id, segments, source, from_job=False, alignment_notes=None
    ):
        from learning_agent.media import MediaError, valid_segments

        info = self.media(space_id, material_id)
        segments = valid_segments(segments, info["duration"])
        with self.connect() as db:
            state = db.execute(
                "SELECT transcript_state FROM media WHERE material_id=?", (material_id,)
            ).fetchone()[0]
            if state == "running" and not from_job:
                raise MediaError("正在转写，请等待结束后再上传字幕，避免相互覆盖")
            db.execute("DELETE FROM transcript_segments WHERE material_id=?", (material_id,))
            self.invalidate_index(db, material_id)
            db.executemany(
                "INSERT INTO transcript_segments VALUES (?,?,?,?,?)",
                [(material_id, i, s["start"], s["end"], s["text"]) for i, s in enumerate(segments)],
            )
            db.execute(
                "UPDATE media SET transcript_state='ready',transcript_source=?,transcript_error=NULL WHERE material_id=?",
                (source, material_id),
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
            db.execute("UPDATE materials SET source=? WHERE id=?", (json.dumps(source), material_id))
            self.invalidate_index(db, material_id)
        return source

    def transcribe_media(self, space_id, material_id, settings):
        from learning_agent.media import MediaError, transcribe

        info = self.media(space_id, material_id)
        try:
            result = transcribe(self.root / info["original_path"], info["duration"], settings)
            self.save_segments(
                space_id,
                material_id,
                result["segments"],
                f"asr:{settings.asr_provider}:{settings.asr_model}",
                from_job=True,
                alignment_notes=result["alignment_notes"],
            )
        except MediaError as error:
            with self.connect() as db:
                db.execute(
                    "UPDATE media SET transcript_state='failed',transcript_error=? WHERE material_id=?",
                    (str(error), material_id),
                )

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
                    "SELECT id,space_id FROM conversations WHERE space_id=? ORDER BY rowid DESC",
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

    def start_run(self, conversation_id, content):
        run_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs(id,conversation_id,state) VALUES (?,?,'running')",
                (run_id, conversation_id),
            )
            db.execute(
                "INSERT INTO messages(conversation_id,role,content,run_id) VALUES (?,'user',?,?)",
                (conversation_id, content, run_id),
            )
        return run_id

    def save_events(self, run_id, events):
        with self.connect() as db:
            db.execute("UPDATE runs SET events=? WHERE id=?", (json.dumps(events), run_id))

    def finish_run(self, run_id, state, content, protocol, error=None):
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
