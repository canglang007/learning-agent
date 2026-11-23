"""Durable single-user jobs. Network and media work never run in these transactions."""

import json
import time
from uuid import uuid4


class JobConflict(ValueError):
    pass


class JobStopped(Exception):
    pass


def configuration(kind, settings):
    if kind == "preview":
        return {"pipeline": "preview-v1"}
    if kind == "ocr":
        return {
            "pipeline": "page-ocr-v1",
            "min_text_characters": settings.ocr_min_text_characters,
        }
    if kind == "transcribe":
        return {
            "pipeline": "parts-transcript-v2",
            "provider": settings.asr_provider,
            "model": settings.asr_model,
            "base_url": settings.asr_base_url.rstrip("/"),
            "part_seconds": settings.asr_local_part_seconds
            if settings.asr_provider == "local"
            else settings.asr_api_part_seconds,
            "context_seconds": settings.asr_context_seconds,
        }
    return {
        "pipeline": "embedding-v1",
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "base_url": ""
        if settings.embedding_provider == "local"
        else settings.embedding_base_url.rstrip("/"),
    }


class Jobs:
    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, space_id TEXT NOT NULL, material_id TEXT,
                    target TEXT NOT NULL, kind TEXT NOT NULL, input_version TEXT NOT NULL,
                    config TEXT NOT NULL, state TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'queued',
                    completed INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 1,
                    attempt INTEGER NOT NULL DEFAULT 0, token TEXT,
                    pause_requested INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    remote_pending INTEGER NOT NULL DEFAULT 0, error TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs(target,kind)
                    WHERE state IN ('queued','running','paused');
                CREATE TABLE IF NOT EXISTS job_parts (
                    job_id TEXT NOT NULL, part TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY(job_id,part));
                CREATE TABLE IF NOT EXISTS worker_status (
                    id INTEGER PRIMARY KEY CHECK(id=1), pid INTEGER NOT NULL,
                    heartbeat REAL NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
            for column in ("processed_seconds", "total_seconds"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} REAL NOT NULL DEFAULT 0")

    def version(self, kind, space, material=None, db=None):
        if db is None:
            with self.store.connect() as connection:
                return self.version(kind, space, material, connection)
        if kind == "index":
            return json.dumps(
                [
                    list(r)
                    for r in db.execute(
                        "SELECT id,revision FROM materials WHERE space_id=? ORDER BY id", (space,)
                    )
                ],
                separators=(",", ":"),
            )
        if kind == "ocr":
            # The input of OCR is the set of pages, not the extracted text: the
            # text changes *because of* this job, so keying on it would abort
            # the job on its own first write.
            row = db.execute(
                "SELECT count(*) FROM material_pages WHERE material_id=?", (material,)
            ).fetchone()
            return f"pages:{row[0]}"
        row = db.execute(
            "SELECT metadata,transcript_revision FROM media WHERE material_id=?", (material,)
        ).fetchone()
        return (
            str(row["transcript_revision"])
            if kind == "transcribe"
            else json.loads(row["metadata"])["original_path"]
        )

    def enqueue(self, kind, space, settings, material=None):
        self.store.require_space(space)
        if material:
            if kind == "ocr":
                if self.store.material(space, material)["format"] != "pdf":
                    raise JobConflict("页面文字识别只适用于 PDF")
            else:
                info = self.store.media(space, material)
                if kind == "transcribe" and not info["has_audio"]:
                    from learning_agent.media import MediaError

                    raise MediaError("该视频没有音轨，可查看画面或上传已有字幕")
        config = json.dumps(configuration(kind, settings), sort_keys=True)
        target = material or space
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            version = self.version(kind, space, material, db)
            old = db.execute(
                "SELECT * FROM jobs WHERE target=? AND kind=? AND state IN ('queued','running','paused')",
                (target, kind),
            ).fetchone()
            if old and old["input_version"] == version and old["config"] == config:
                return dict(old)
            if old:
                db.execute(
                    "UPDATE jobs SET state='cancelled',cancel_requested=1,updated=? WHERE id=?",
                    (time.time(), old["id"]),
                )
            reusable = db.execute(
                "SELECT id FROM jobs WHERE target=? AND kind=? AND input_version=? AND config=? AND state='failed' ORDER BY created DESC LIMIT 1",
                (target, kind, version, config),
            ).fetchone()
            if reusable:
                # Calling the processing endpoint again is an explicit retry, with saved parts.
                db.execute(
                    "UPDATE jobs SET state='queued',pause_requested=0,cancel_requested=0,remote_pending=0,error=NULL,updated=? WHERE id=?",
                    (time.time(), reusable[0]),
                )
                self.project(db, reusable[0])
                return dict(db.execute("SELECT * FROM jobs WHERE id=?", (reusable[0],)).fetchone())
            identifier, now = str(uuid4()), time.time()
            db.execute(
                "INSERT INTO jobs(id,space_id,material_id,target,kind,input_version,config,state,created,updated) VALUES (?,?,?,?,?,?,?,'queued',?,?)",
                (identifier, space, material, target, kind, version, config, now, now),
            )
            self.project(db, identifier)
        return self.get(identifier)

    def get(self, identifier):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise LookupError("任务不存在")
        return dict(row)

    def list(self, space):
        self.store.require_space(space)
        with self.store.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM jobs WHERE space_id=? ORDER BY created DESC LIMIT 50", (space,)
                )
            ]

    def online(self):
        with self.store.connect() as db:
            row = db.execute("SELECT heartbeat FROM worker_status WHERE id=1").fetchone()
        return bool(row and time.time() - row[0] < 10)

    def claim(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET state='running',stage='starting',attempt=attempt+1,token=?,updated=? WHERE id=?",
                (str(uuid4()), time.time(), row[0]),
            )
            self.project(db, row[0])
        return self.get(row[0])

    def check(self, job, db):
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if row["state"] != "running" or row["token"] != job["token"]:
            raise JobStopped()
        if (
            self.version(row["kind"], row["space_id"], row["material_id"], db)
            != row["input_version"]
        ):
            # Caller catches JobStopped; state changes are committed outside the raising transaction.
            raise JobStopped()
        if row["pause_requested"] or row["cancel_requested"]:
            raise JobStopped()

    def checkpoint(
        self, job, stage, completed=None, total=None, remote=False, seconds=None, duration=None
    ):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.check(job, db)
            db.execute(
                "UPDATE jobs SET stage=?,completed=coalesce(?,completed),total=coalesce(?,total),remote_pending=?,processed_seconds=coalesce(?,processed_seconds),total_seconds=coalesce(?,total_seconds),updated=? WHERE id=?",
                (stage, completed, total, int(remote), seconds, duration, time.time(), job["id"]),
            )

    def part(self, job, key):
        with self.store.connect() as db:
            row = db.execute(
                "SELECT result FROM job_parts WHERE job_id=? AND part=?", (job["id"], key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_part(self, job, key, result, db=None):
        if db is None:
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                return self.save_part(job, key, result, connection)
        # A pause does not discard a just-completed response. Publication checks it separately.
        row = db.execute("SELECT state,token FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if row["state"] != "running" or row["token"] != job["token"]:
            raise JobStopped()
        db.execute(
            "INSERT OR REPLACE INTO job_parts VALUES (?,?,?)",
            (
                job["id"],
                key,
                json.dumps(result, ensure_ascii=False),
            ),
        )
        db.execute(
            "UPDATE jobs SET remote_pending=0,updated=? WHERE id=?", (time.time(), job["id"])
        )

    def finish(self, job, db):
        self.check(job, db)
        db.execute(
            "UPDATE jobs SET state='succeeded',stage='completed',completed=total,remote_pending=0,error=NULL,updated=? WHERE id=?",
            (time.time(), job["id"]),
        )
        self.project(db, job["id"])

    def stop(self, job, error=None):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if row["state"] != "running" or row["token"] != job["token"]:
                return
            stale = (
                self.version(row["kind"], row["space_id"], row["material_id"], db)
                != row["input_version"]
            )
            state = (
                "cancelled"
                if stale or row["cancel_requested"]
                else ("paused" if row["pause_requested"] else "failed")
            )
            if row["remote_pending"]:
                error = "remote_result_unknown：远端结果未保存，重试可能再次计费"
                if state == "paused":
                    state = "failed"
            db.execute(
                "UPDATE jobs SET state=?,error=?,updated=? WHERE id=?",
                (
                    state,
                    error,
                    time.time(),
                    job["id"],
                ),
            )
            self.project(db, job["id"])

    def action(self, identifier, action):
        if action not in {"pause", "resume", "retry", "cancel"}:
            raise JobConflict("不支持的任务操作")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise LookupError("任务不存在")
            state = row["state"]
            if action == "cancel" and state in {"queued", "running", "paused", "failed"}:
                # Publish guards invalidate immediately; in-flight work may still need to return.
                db.execute(
                    "UPDATE jobs SET state='cancelled',cancel_requested=1 WHERE id=?", (identifier,)
                )
            elif action == "pause" and state in {"queued", "running"}:
                db.execute(
                    "UPDATE jobs SET pause_requested=1,state=? WHERE id=?",
                    (
                        "paused" if state == "queued" else state,
                        identifier,
                    ),
                )
            elif (action == "resume" and state == "paused") or (
                action == "retry" and state == "failed"
            ):
                if (
                    self.version(row["kind"], row["space_id"], row["material_id"], db)
                    != row["input_version"]
                ):
                    raise JobConflict("资料版本已变化，请重新创建任务")
                other = db.execute(
                    "SELECT 1 FROM jobs WHERE target=? AND kind=? AND state IN ('queued','running','paused') AND id!=?",
                    (row["target"], row["kind"], identifier),
                ).fetchone()
                if other:
                    raise JobConflict("已有更新任务，请使用该任务")
                db.execute(
                    "UPDATE jobs SET state='queued',pause_requested=0,cancel_requested=0,remote_pending=0,error=NULL WHERE id=?",
                    (identifier,),
                )
            else:
                raise JobConflict("当前任务状态不支持此操作")
            db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time(), identifier))
            self.project(db, identifier)
        return self.get(identifier)

    @staticmethod
    def project(db, identifier):
        row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if not row["material_id"]:
            return
        if row["kind"] == "ocr":
            # Per-page text lives in material_pages and progress lives in the job
            # row; there is no media record to project onto.
            return
        if row["kind"] == "preview":
            # A queued retry must not hide an already published usable preview.
            db.execute(
                "UPDATE media SET state=?,error=? WHERE material_id=? AND state!='ready'",
                (
                    "processing"
                    if row["state"] in {"queued", "running"}
                    else "ready"
                    if row["state"] == "succeeded"
                    else row["state"],
                    row["error"],
                    row["material_id"],
                ),
            )
        else:
            db.execute(
                "UPDATE media SET transcript_state=?,transcript_error=? WHERE material_id=?",
                (
                    "running"
                    if row["state"] in {"queued", "running"}
                    else "ready"
                    if row["state"] == "succeeded"
                    else row["state"],
                    row["error"],
                    row["material_id"],
                ),
            )

    def recover(self):
        """Only called while the worker's OS lock is held, never by API startup."""
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM jobs WHERE state='running'").fetchall():
                state = (
                    "failed"
                    if row["remote_pending"]
                    else ("paused" if row["pause_requested"] else "queued")
                )
                error = (
                    "remote_result_unknown：远端结果未保存，重试可能再次计费"
                    if row["remote_pending"]
                    else None
                )
                db.execute(
                    "UPDATE jobs SET state=?,error=?,token=NULL,updated=? WHERE id=?",
                    (
                        state,
                        error,
                        time.time(),
                        row["id"],
                    ),
                )
                self.project(db, row["id"])
