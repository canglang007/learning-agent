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
            """)

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
            db.execute("INSERT INTO materials VALUES (:id,:space_id,:name,:text)", item)
        return {key: value for key, value in item.items() if key != "text"}

    def materials(self, space_id):
        self.require_space(space_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,name FROM materials WHERE space_id=? ORDER BY rowid", (space_id,)
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
