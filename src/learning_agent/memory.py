"""Explicit, scoped learning memory. Never mine chats for durable facts automatically."""

import json
import re
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from learning_agent.learning import LearningConflict
from learning_agent.run_store import PENDING_SQL


class TopicCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=120)
    material_ids: list[str] = Field(default_factory=list, max_length=50)


class ContextSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    topic_id: str | None = None
    memory_enabled: bool = False


class MemoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    topic_id: str
    kind: str = Field(pattern="^(preference|goal|unresolved)$")
    text: str = Field(min_length=1, max_length=600)
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    source_run_id: str | None = None

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value:
            raise ValueError("记忆内容不能为空")
        return value


class MemoryEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    revision: int = Field(ge=1)
    text: str | None = Field(default=None, min_length=1, max_length=600)
    status: str | None = Field(default=None, pattern="^(active|disabled)$")


def terms(text):
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    for span in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(span[i : i + 2] for i in range(len(span) - 1))
    return words


class MemoryStore:
    def init_memory(self):
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS learning_topics (
                    id TEXT PRIMARY KEY, space_id TEXT NOT NULL, title TEXT NOT NULL,
                    material_ids TEXT NOT NULL, created_at TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS context_settings (
                    conversation_id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0,
                    topic_id TEXT, memory_enabled INTEGER NOT NULL DEFAULT 0,
                    history_after INTEGER NOT NULL DEFAULT -1);
                CREATE TABLE IF NOT EXISTS learning_memories (
                    id TEXT PRIMARY KEY, space_id TEXT NOT NULL, topic_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS memory_versions (
                    memory_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY(memory_id,revision));
            """)
            columns = {r[1] for r in db.execute("PRAGMA table_info(context_settings)")}
            if "material_versions" not in columns:
                db.execute(
                    "ALTER TABLE context_settings ADD COLUMN material_versions TEXT NOT NULL DEFAULT '{}'"
                )
            # 主题与记忆原先整个空间共享；加 owner 后各账号只见自己的。
            # 老行的 owner 为空，不再归任何人，等于下线。
            for table in ("learning_topics", "learning_memories"):
                columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                if "owner" not in columns:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN owner TEXT NOT NULL DEFAULT ''")

    def learning_topics(self, space_id, owner=""):
        """列学习主题；``owner`` 是归属键，只返回自己的和公共的（``owner=''``）。"""
        self.require_space(space_id)
        with self.connect() as db:
            return [
                {**dict(r), "material_ids": json.loads(r["material_ids"])}
                for r in db.execute(
                    "SELECT id,space_id,title,material_ids,created_at FROM learning_topics "
                    "WHERE space_id=? AND (owner='' OR owner=?)",
                    (space_id, owner),
                )
            ]

    def learning_topic(self, space_id, topic_id, owner=""):
        for topic in self.learning_topics(space_id, owner):
            if topic["id"] == topic_id:
                return topic
        raise LookupError("当前资料空间没有此学习主题")

    def create_learning_topic(self, space_id, payload, owner=""):
        self.require_space(space_id)
        ids = list(dict.fromkeys(payload["material_ids"]))
        for mid in ids:
            self.material(space_id, mid)
        topic = {
            "id": str(uuid4()),
            "space_id": space_id,
            "title": payload["title"],
            "material_ids": ids,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO learning_topics(id,space_id,title,material_ids,created_at,owner) "
                "VALUES (?,?,?,?,?,?)",
                (topic["id"], space_id, topic["title"], json.dumps(ids), topic["created_at"], owner),
            )
        return topic

    def context_settings(self, conversation_id):
        self.get_conversation(conversation_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM context_settings WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
        result = (
            dict(row)
            if row
            else {
                "conversation_id": conversation_id,
                "revision": 0,
                "topic_id": None,
                "memory_enabled": False,
                "history_after": -1,
            }
        )
        result.pop("material_versions", None)
        return result

    def refresh_memory_versions(self, conversation_id):
        """Invalidate derived history on the next turn after material replacement."""
        conv = self.get_conversation(conversation_id)
        setting = self.context_settings(conversation_id)
        if not setting["memory_enabled"]:
            return
        topic = self.learning_topic(conv["space_id"], setting["topic_id"], conv.get("owner", ""))
        current = json.dumps(
            {
                mid: self.material(conv["space_id"], mid)["revision"]
                for mid in topic["material_ids"]
            },
            sort_keys=True,
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT material_versions FROM context_settings WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()[0]
            if prior != current:
                self._idle(db, [conversation_id])
                self._invalidate_memory_context(db, [conversation_id])
                db.execute(
                    "UPDATE context_settings SET material_versions=? WHERE conversation_id=?",
                    (current, conversation_id),
                )

    @staticmethod
    def _idle(db, conversations):
        for cid in conversations:
            if db.execute(
                f"SELECT 1 FROM runs WHERE conversation_id=? AND state IN {PENDING_SQL}", (cid,)
            ).fetchone():
                raise LearningConflict("相关会话正在回答，请完成后再修改主题或记忆")

    @staticmethod
    def _invalidate_memory_context(db, conversations):
        # Old answers/tool results can echo a revoked memory. Keep the raw archive,
        # but stop automatic replay before this boundary, including derived notes.
        for cid in conversations:
            end = db.execute(
                "SELECT COALESCE(MAX(turn_seq),0) FROM protocol_turns WHERE conversation_id=?",
                (cid,),
            ).fetchone()[0]
            db.execute(
                "UPDATE context_settings SET history_after=? WHERE conversation_id=?", (end, cid)
            )
            row = db.execute(
                "SELECT state_json FROM learning_states WHERE conversation_id=?", (cid,)
            ).fetchone()
            if row:
                state = json.loads(row[0])
                for key in ("goal", "active_topic", "confusion"):
                    if state[key] and state[key]["origin"] != "user_explicit":
                        state[key] = None
                if not state["goal"]:
                    state["goal_pinned"] = False
                state["open_questions"] = [
                    q for q in state["open_questions"] if q["origin"] == "user_explicit"
                ]
                state.update(attempts=[], evidence_refs=[], reset=True)
                state["revision"] += 1
                state["context_epoch"] += 1
                db.execute(
                    "UPDATE learning_states SET revision=?,state_json=?,updated_at=? WHERE conversation_id=?",
                    (state["revision"], json.dumps(state), datetime.now(UTC).isoformat(), cid),
                )
            db.execute(
                "UPDATE context_summaries SET status='invalidated' WHERE conversation_id=? AND status IN ('ready','running')",
                (cid,),
            )

    def edit_context_settings(self, conversation_id, payload):
        conv = self.get_conversation(conversation_id)
        if payload["memory_enabled"] and not payload["topic_id"]:
            raise ValueError("请先选择要接续的学习主题")
        if payload["topic_id"]:
            self.learning_topic(conv["space_id"], payload["topic_id"], conv.get("owner", ""))
        self.current_learning_state(conversation_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._idle(db, [conversation_id])
            row = db.execute(
                "SELECT * FROM context_settings WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            if (row["revision"] if row else 0) != payload["revision"]:
                raise LearningConflict("接续设置已更新，请刷新后重试")
            if (row["topic_id"] if row else None) == payload["topic_id"] and bool(
                row["memory_enabled"] if row else False
            ) == payload["memory_enabled"]:
                return self.context_settings(conversation_id)
            db.execute(
                "INSERT INTO context_settings(conversation_id,revision,topic_id,memory_enabled) VALUES (?,?,?,?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET revision=excluded.revision,topic_id=excluded.topic_id,memory_enabled=excluded.memory_enabled",
                (
                    conversation_id,
                    payload["revision"] + 1,
                    payload["topic_id"],
                    payload["memory_enabled"],
                ),
            )
            self._invalidate_memory_context(db, [conversation_id])
        return self.context_settings(conversation_id)

    def memories(self, space_id, topic_id, owner=""):
        self.learning_topic(space_id, topic_id, owner)
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM learning_memories WHERE space_id=? AND topic_id=? AND (owner='' OR owner=?) AND status!='deleted' ORDER BY rowid",
                    (space_id, topic_id, owner),
                )
            ]

    @staticmethod
    def _memory_dependents(db, topic_id):
        return [
            r[0]
            for r in db.execute(
                "SELECT conversation_id FROM context_settings WHERE topic_id=? AND memory_enabled=1",
                (topic_id,),
            )
        ]

    def create_memory(self, space_id, payload, owner=""):
        topic = self.learning_topic(space_id, payload["topic_id"], owner)
        if not set(payload["material_ids"]) <= set(topic["material_ids"]):
            raise ValueError("记忆资料范围必须属于所选主题；主题未绑定资料时请使用主题级记忆")
        source = None
        if payload["source_run_id"]:
            run = self.run(payload["source_run_id"])
            conv = self.get_conversation(run["conversation_id"])
            setting = self.context_settings(conv["id"])
            if (
                conv["space_id"] != space_id
                or conv.get("owner", "") != owner
                or setting["topic_id"] != topic["id"]
                or run["state"] not in ("completed", "budget_exhausted")
            ):
                raise ValueError("来源必须是本空间、所选主题的已完成回答")
            source = {"run_id": run["id"], "conversation_id": conv["id"]}
            observed = {
                e["observation"]["material_id"]: e["material_revision"]
                for e in run["events"]
                if e.get("type") == "tool"
                and e.get("material_revision") is not None
                and e.get("observation", {}).get("material_id")
                and not e["observation"].get("error")
            }
            if not set(observed) <= set(topic["material_ids"]):
                raise ValueError("来源回答使用了主题范围外的资料，请创建包含这些资料的主题")
        memory = dict(
            id=str(uuid4()),
            space_id=space_id,
            **payload,
            revision=1,
            status="active",
            origin="user_confirmed" if source else "user_explicit",
            source=source,
            updated_at=datetime.now(UTC).isoformat(),
        )
        memory["material_revisions"] = {
            mid: self.material(space_id, mid)["revision"] for mid in payload["material_ids"]
        }
        if source:
            memory["material_revisions"].update(observed)
        memory.pop("source_run_id")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            dependents = self._memory_dependents(db, topic["id"])
            self._idle(db, dependents)
            if (
                db.execute(
                    "SELECT COUNT(*) FROM learning_memories WHERE topic_id=? AND status!='deleted'",
                    (topic["id"],),
                ).fetchone()[0]
                >= 100
            ):
                raise ValueError("每个主题最多保存100条记忆，请整理后再新增")
            db.execute(
                "INSERT INTO learning_memories(id,space_id,topic_id,revision,payload,status,owner) "
                "VALUES (?,?,?,?,?,?,?)",
                (memory["id"], space_id, topic["id"], 1, json.dumps(memory), "active", owner),
            )
            db.execute(
                "INSERT INTO memory_versions VALUES (?,?,?,?)",
                (memory["id"], 1, json.dumps(memory), memory["updated_at"]),
            )
            self._invalidate_memory_context(db, dependents)
        return memory

    def edit_memory(self, space_id, memory_id, payload, *, forget=False, owner=""):
        self.require_space(space_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM learning_memories WHERE id=? AND space_id=? AND (owner='' OR owner=?) AND status!='deleted'",
                (memory_id, space_id, owner),
            ).fetchone()
            if not row:
                raise LookupError("当前空间没有此记忆")
            if row["revision"] != payload["revision"]:
                raise LearningConflict("记忆已更新，请刷新后重试")
            dependents = self._memory_dependents(db, row["topic_id"])
            self._idle(db, dependents)
            prior = json.loads(row["payload"])
            memory = {
                **prior,
                **{k: v for k, v in payload.items() if v is not None},
                "revision": row["revision"] + 1,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            # Editing/re-enabling explicitly reconfirms the entry against current versions.
            memory["material_revisions"] = {
                mid: self.material(space_id, mid)["revision"] for mid in prior["material_revisions"]
            }
            if forget:
                # Erase version contents too, retaining only an ID/revision tombstone.
                memory = {
                    k: memory[k] for k in ("id", "space_id", "topic_id", "revision", "updated_at")
                }
                memory["status"] = "deleted"
                db.execute("DELETE FROM memory_versions WHERE memory_id=?", (memory_id,))
                for frozen in db.execute(
                    "SELECT i.run_id,i.payload FROM run_inputs i JOIN runs r ON r.id=i.run_id "
                    "JOIN conversations c ON c.id=r.conversation_id WHERE c.space_id=? AND i.payload LIKE ?",
                    (space_id, "%" + memory_id + "%"),
                ).fetchall():
                    payload = json.loads(frozen["payload"])
                    bundle = payload.get("memories", {})
                    bundle["items"] = [m for m in bundle.get("items", []) if m["id"] != memory_id]
                    bundle.setdefault("redacted_memory_ids", []).append(memory_id)
                    db.execute(
                        "UPDATE run_inputs SET payload=? WHERE run_id=?",
                        (json.dumps(payload), frozen["run_id"]),
                    )
                    # Terminal execution snapshots contain derived injected messages too.
                    db.execute("DELETE FROM run_checkpoints WHERE run_id=?", (frozen["run_id"],))
                for turn in db.execute(
                    "SELECT p.conversation_id,p.turn_seq,p.raw_input FROM protocol_turns p "
                    "JOIN conversations c ON c.id=p.conversation_id WHERE c.space_id=? "
                    "AND p.raw_input LIKE ?",
                    (space_id, "%" + memory_id + "%"),
                ).fetchall():
                    raw = json.loads(turn["raw_input"])
                    bundle = raw.get("derived_context", {}).get("memories", {})
                    bundle["items"] = [m for m in bundle.get("items", []) if m["id"] != memory_id]
                    bundle.setdefault("redacted_memory_ids", []).append(memory_id)
                    db.execute(
                        "UPDATE protocol_turns SET raw_input=? WHERE conversation_id=? AND turn_seq=?",
                        (json.dumps(raw), turn["conversation_id"], turn["turn_seq"]),
                    )
            db.execute(
                "UPDATE learning_memories SET revision=?,payload=?,status=? WHERE id=?",
                (memory["revision"], json.dumps(memory), memory["status"], memory_id),
            )
            db.execute(
                "INSERT INTO memory_versions VALUES (?,?,?,?)",
                (memory_id, memory["revision"], json.dumps(memory), memory["updated_at"]),
            )
            self._invalidate_memory_context(db, dependents)
        return memory

    def recall_memories(self, conversation_id, question, focus=None):
        conv, setting = (
            self.get_conversation(conversation_id),
            self.context_settings(conversation_id),
        )
        bundle = {
            "enabled": bool(setting["memory_enabled"]),
            "settings_revision": setting["revision"],
            "items": [],
        }
        if not setting["memory_enabled"]:
            return bundle
        topic = self.learning_topic(conv["space_id"], setting["topic_id"], conv.get("owner", ""))
        bundle["topic"] = topic
        # A focused source outside the chosen topic is a different learning scope.
        mid = focus.get("material_id") if focus else None
        if mid and mid not in topic["material_ids"]:
            bundle["omission"] = "当前资料位置不属于接续主题，未召回记忆"
            return bundle
        query = terms(question)
        candidates = []
        for memory in self.memories(conv["space_id"], topic["id"], conv.get("owner", "")):
            if memory["status"] != "active":
                continue
            if any(
                self.material(conv["space_id"], mid)["revision"] != rev
                for mid, rev in memory.get("material_revisions", {}).items()
            ):
                continue
            if memory["material_ids"] and (not mid or mid not in memory["material_ids"]):
                continue
            overlap = len(query & terms(memory["text"]))
            if memory["kind"] == "unresolved" and not overlap:
                continue
            candidates.append((memory["kind"] != "unresolved", overlap, memory))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]["updated_at"]), reverse=True)
        chars = 0
        for _, _, memory in candidates:
            if len(bundle["items"]) >= 6 or chars + len(memory["text"]) > 1800:
                continue
            refs = [
                {"material_id": m, "version_status": "must_recheck"} for m in memory["material_ids"]
            ]
            bundle["items"].append({**memory, "source_pointers": refs})
            chars += len(memory["text"])
        return bundle
