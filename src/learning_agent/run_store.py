"""Durable RT1 submission, control, and public progress; no mid-run replay yet."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from learning_agent.learning import FeedbackError, LearningConflict

PENDING_SQL = "('queued','running','cancelling','interrupted')"
SUCCESS = {"completed", "budget_exhausted"}
TERMINAL = {*SUCCESS, "failed", "cancelled"}


class RuntimeConflict(LearningConflict):
    pass


class RuntimeUnavailable(Exception):
    pass


class RunStopped(Exception):
    def __init__(self, interrupted=False):
        self.interrupted = interrupted


def now():
    return datetime.now(UTC).isoformat()


class RunStore:
    def init_runtime(self):
        with self.connect() as db:
            columns = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
            for name, declaration in {
                "client_request_id": "TEXT",
                "state_revision": "INTEGER NOT NULL DEFAULT 0",
                "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                "owner_token": "TEXT",
                "phase": "TEXT NOT NULL DEFAULT ''",
                "created_at": "TEXT",
                "updated_at": "TEXT",
                "recovery_reason": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")
            if "kind" not in {r[1] for r in db.execute("PRAGMA table_info(messages)")}:
                db.execute("ALTER TABLE messages ADD COLUMN kind TEXT NOT NULL DEFAULT 'answer'")
                db.execute(
                    "UPDATE messages SET kind='system' WHERE role='assistant' AND run_id IN (SELECT id FROM runs WHERE state NOT IN ('completed','budget_exhausted'))"
                )
            db.executescript(f"""
                CREATE TABLE IF NOT EXISTS run_inputs (
                    run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(run_id,seq));
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    run_id TEXT PRIMARY KEY, phase TEXT NOT NULL, payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS run_operations (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, purpose TEXT NOT NULL,
                    step INTEGER NOT NULL, state TEXT NOT NULL, result TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS run_request_identity
                    ON runs(conversation_id,client_request_id) WHERE client_request_id IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS one_run_message
                    ON messages(run_id,role) WHERE run_id IS NOT NULL;
                DROP INDEX IF EXISTS one_running_turn;
                CREATE UNIQUE INDEX one_running_turn ON runs(conversation_id) WHERE state IN {PENDING_SQL};
            """)

    @staticmethod
    def event_in(db, run_id, event_type, payload=None):
        seq = db.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM run_events WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO run_events VALUES (?,?,?,?,?)",
            (run_id, seq, event_type, json.dumps(payload or {}, ensure_ascii=False), now()),
        )
        return seq

    def progress(self, run_id, owner, event_type, payload=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner_token,state,cancel_requested FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if (
                not row
                or row["owner_token"] != owner
                or row["state"] != "running"
                or row["cancel_requested"]
            ):
                raise RunStopped()
            self.event_in(db, run_id, event_type, payload)
            db.execute(
                "UPDATE runs SET phase=?,updated_at=? WHERE id=?", (event_type, now(), run_id)
            )

    def run_input(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM run_inputs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise LookupError("任务没有可用的提交记录")
        return json.loads(row[0])

    def submit_run(self, conversation_id, payload, profile, queue_limit):
        conv = self.get_conversation(conversation_id)
        request_id = payload["client_request_id"]
        identity = {k: payload.get(k) for k in ("content", "focus", "feedback")}
        prior = self.existing_submission(conversation_id, request_id, identity)
        if prior:
            return prior
        with self.connect() as db:
            if db.execute(
                f"SELECT 1 FROM runs WHERE conversation_id=? AND state IN {PENDING_SQL}",
                (conversation_id,),
            ).fetchone():
                raise RuntimeConflict("本会话有待处理问答，请完成或结束此轮后再提问")
        self.refresh_memory_versions(conversation_id)
        self.context_turns(conversation_id)
        learning = self.learning_state(conversation_id)
        settings = self.context_settings(conversation_id)
        memories = self.recall_memories(conversation_id, payload["content"], payload.get("focus"))
        feedback = payload.get("feedback")
        target = (
            self.feedback_target(conversation_id, feedback["target_run_id"]) if feedback else None
        )
        frozen = {
            "identity": identity,
            "learning": learning,
            "context_settings": settings,
            "memories": memories,
            "target": target,
            "profile": profile,
            "space_id": conv["space_id"],
            "schema_version": 1,
        }
        run_id = str(uuid4())
        return self._insert_submission(
            conversation_id, request_id, identity, frozen, run_id, queue_limit
        )

    def existing_submission(self, conversation_id, request_id, identity):
        with self.connect() as db:
            prior = db.execute(
                "SELECT id FROM runs WHERE conversation_id=? AND client_request_id=?",
                (conversation_id, request_id),
            ).fetchone()
        if prior:
            if self.run_input(prior[0])["identity"] != identity:
                raise RuntimeConflict("相同提交ID对应不同问题，请刷新或使用新的提交ID")
            return prior[0]
        return None

    def _insert_submission(
        self, conversation_id, request_id, identity, frozen, run_id, queue_limit
    ):
        learning, feedback = frozen["learning"], identity["feedback"]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT id FROM runs WHERE conversation_id=? AND client_request_id=?",
                (conversation_id, request_id),
            ).fetchone()
            if prior:
                saved = json.loads(
                    db.execute(
                        "SELECT payload FROM run_inputs WHERE run_id=?", (prior[0],)
                    ).fetchone()[0]
                )
                if saved["identity"] != identity:
                    raise RuntimeConflict("相同提交ID对应不同问题")
                return prior[0]
            if db.execute(
                f"SELECT 1 FROM runs WHERE conversation_id=? AND state IN {PENDING_SQL}",
                (conversation_id,),
            ).fetchone():
                raise RuntimeConflict("本会话有待处理问答，请先完成或结束此轮")
            current = db.execute(
                "SELECT revision FROM learning_states WHERE conversation_id=?", (conversation_id,)
            ).fetchone()[0]
            if current != learning["revision"]:
                raise RuntimeConflict("学习状态刚刚更新，请重新提交")
            if (
                db.execute("SELECT count(*) FROM runs WHERE state='queued'").fetchone()[0]
                >= queue_limit
            ):
                raise RuntimeUnavailable("问答队列已满，请稍后重试；尚未保存本次问题")
            if feedback:
                source = db.execute(
                    "SELECT conversation_id,state FROM runs WHERE id=?",
                    (feedback["target_run_id"],),
                ).fetchone()
                if (
                    not source
                    or source["conversation_id"] != conversation_id
                    or source["state"] not in SUCCESS
                ):
                    raise FeedbackError("反馈需要指向当前会话的已完成讲解")
            db.execute(
                "INSERT INTO runs(id,conversation_id,state,client_request_id,state_revision,phase,created_at,updated_at) VALUES (?,?,'queued',?,1,'queued',?,?)",
                (run_id, conversation_id, request_id, now(), now()),
            )
            db.execute(
                "INSERT INTO messages(conversation_id,role,content,run_id) VALUES (?,'user',?,?)",
                (conversation_id, identity["content"], run_id),
            )
            if feedback:
                db.execute(
                    "INSERT INTO learning_feedback VALUES (?,?,?,?)",
                    (run_id, feedback["target_run_id"], feedback["kind"], feedback["focus_text"]),
                )
            db.execute(
                "INSERT INTO run_inputs VALUES (?,?)",
                (run_id, json.dumps(frozen, ensure_ascii=False)),
            )
            self.event_in(db, run_id, "run.accepted")
        return run_id

    def claim_run(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM runs WHERE state='queued' AND client_request_id IS NOT NULL ORDER BY rowid LIMIT 1"
            ).fetchone()
            if not row:
                return None
            owner = str(uuid4())
            db.execute(
                "UPDATE runs SET state='running',owner_token=?,state_revision=state_revision+1,phase='starting',updated_at=? WHERE id=?",
                (owner, now(), row[0]),
            )
            self.event_in(db, row[0], "run.started")
        return row[0], owner

    def operation(self, run_id, owner, purpose, step, *, identifier=None, result=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner_token,state,cancel_requested FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if (
                not row
                or row["owner_token"] != owner
                or row["state"] != "running"
                or row["cancel_requested"]
            ):
                raise RunStopped()
            stamp = now()
            if identifier is None:
                identifier = str(uuid4())
                db.execute(
                    "INSERT INTO run_operations VALUES (?,?,?,?,'prepared',NULL,?,?)",
                    (identifier, run_id, purpose, step, stamp, stamp),
                )
                phase = "prepared"
            else:
                phase = "returned" if result is not None else "sending"
                db.execute(
                    "UPDATE run_operations SET state=?,result=?,updated_at=? WHERE id=? AND run_id=?",
                    (
                        phase,
                        json.dumps(result, ensure_ascii=False) if result is not None else None,
                        stamp,
                        identifier,
                        run_id,
                    ),
                )
            db.execute(
                "INSERT INTO run_checkpoints VALUES (?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET phase=excluded.phase,payload=excluded.payload,updated_at=excluded.updated_at",
                (
                    run_id,
                    phase,
                    json.dumps(
                        {
                            "operation_id": identifier,
                            "purpose": purpose,
                            "step": step,
                            "recoverable": False,
                        }
                    ),
                    stamp,
                ),
            )
        return identifier

    def interrupt_run(self, run_id, owner, reason="process_interrupted"):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state,owner_token FROM runs WHERE id=?", (run_id,)).fetchone()
            if (
                not row
                or row["state"] not in {"running", "cancelling"}
                or row["owner_token"] != owner
            ):
                return
            db.execute(
                "UPDATE runs SET state='interrupted',owner_token=NULL,state_revision=state_revision+1,phase='interrupted',recovery_reason=?,updated_at=? WHERE id=?",
                (reason, now(), run_id),
            )
            db.execute(
                "UPDATE run_operations SET state='unknown',updated_at=? WHERE run_id=? AND state='sending'",
                (now(), run_id),
            )
            self.event_in(db, run_id, "run.interrupted", {"reason": reason})

    def recover_runs(self):
        # Only the exclusive lifespan owner calls this; Store construction never recovers.
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,owner_token,cancel_requested FROM runs WHERE state IN ('running','cancelling')"
            ).fetchall()
        for row in rows:
            self.interrupt_run(row["id"], row["owner_token"], "rt1_restart_requires_new_turn")
            if row["cancel_requested"]:
                self.cancel_run(row["id"])

    def cancel_run(self, run_id):
        self.context_turns(self.run(run_id)["conversation_id"])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT rowid AS cursor,* FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                raise LookupError("问答任务不存在")
            if row["state"] in TERMINAL or row["state"] == "cancelling":
                return
            if row["state"] == "running":
                db.execute(
                    "UPDATE runs SET state='cancelling',cancel_requested=1,state_revision=state_revision+1,phase='cancelling',updated_at=? WHERE id=?",
                    (now(), run_id),
                )
                self.event_in(db, run_id, "run.cancel_requested")
                return
            self._seal_stopped(db, row, "cancelled", "本轮已结束，未提交讲解；可以重新提问。")

    def _seal_stopped(self, db, row, state, content):
        cid, rid = row["conversation_id"], row["id"]
        db.execute(
            "UPDATE runs SET state=?,owner_token=NULL,cancel_requested=?,state_revision=state_revision+1,phase=?,error=?,updated_at=? WHERE id=?",
            (state, int(state == "cancelled"), state, content, now(), rid),
        )
        db.execute(
            "UPDATE run_operations SET state='unknown',updated_at=? WHERE run_id=? AND state='sending'",
            (now(), rid),
        )
        db.execute(
            "INSERT INTO messages(conversation_id,role,content,run_id,kind) VALUES (?,'assistant',?,?,'system')",
            (cid, content, rid),
        )
        question = db.execute(
            "SELECT content FROM messages WHERE run_id=? AND role='user'", (rid,)
        ).fetchone()
        if (
            question
            and not db.execute("SELECT 1 FROM protocol_turns WHERE run_id=?", (rid,)).fetchone()
        ):
            seq = db.execute(
                "SELECT COALESCE(MAX(turn_seq),0)+1 FROM protocol_turns WHERE conversation_id=?",
                (cid,),
            ).fetchone()[0]
            raw = {"question": question[0], "answer": content}
            db.execute(
                "INSERT INTO protocol_turns VALUES (?,?,?,1,'{}',?,'[]','failed',?)",
                (cid, seq, rid, json.dumps(raw), now()),
            )
        self.event_in(db, rid, "run." + state, {"detail": content})

    def fail_run(self, run_id, owner):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row and row["state"] not in TERMINAL and row["owner_token"] == owner:
                self._seal_stopped(
                    db,
                    row,
                    "cancelled" if row["cancel_requested"] else "failed",
                    "问答执行未完成，请结束或重新提问；已有记录保留。",
                )

    def public_run(self, run_id):
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT rowid AS cursor,* FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                raise LookupError("问答任务不存在")
            seq = db.execute(
                "SELECT COALESCE(MAX(seq),0) FROM run_events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            answer = db.execute(
                "SELECT content,kind FROM messages WHERE run_id=? AND role='assistant'", (run_id,)
            ).fetchone()
        result = {
            k: row[k]
            for k in (
                "id",
                "conversation_id",
                "state",
                "error",
                "phase",
                "created_at",
                "updated_at",
                "recovery_reason",
                "cursor",
                "client_request_id",
            )
        }
        result.update(
            revision=row["state_revision"],
            last_event_seq=seq,
            events=json.loads(row["events"]),
            answer=answer[0] if answer else None,
            allowed_actions=["cancel"]
            if row["state"] in {"queued", "running", "interrupted"}
            else [],
            recoverable=False,
        )
        return result

    def list_runs(self, conversation_id, active=False, before=None, limit=20):
        self.get_conversation(conversation_id)
        where = f"AND state IN {PENDING_SQL}" if active else ""
        with self.connect() as db:
            rows = db.execute(
                f"SELECT id FROM runs WHERE conversation_id=? {where} AND (? IS NULL OR rowid<?) ORDER BY rowid DESC LIMIT ?",
                (conversation_id, before, before, limit),
            ).fetchall()
        return [self.public_run(r[0]) for r in rows]

    def progress_events(self, run_id, after=0, limit=100):
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone():
                raise LookupError("问答任务不存在")
            high = db.execute(
                "SELECT COALESCE(MAX(seq),0) FROM run_events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            if after < 0 or after > high:
                raise ValueError("事件游标超出此任务范围")
            return [
                {**dict(r), "payload": json.loads(r["payload"])}
                for r in db.execute(
                    "SELECT * FROM run_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                    (run_id, after, limit),
                )
            ]
