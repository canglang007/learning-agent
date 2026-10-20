import json
from contextlib import contextmanager
from pathlib import Path

from learning_agent.provider import ModelError
from learning_agent.run_store import now


class EvaluationBudgetExceeded(ModelError):
    pass


class Ledger:
    def __init__(self, path, *, readonly=False):
        self.readonly = readonly
        self.path = Path(path)
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, purpose TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        from learning_agent.database import connection

        target = self.path.resolve().as_uri() + "?mode=ro" if self.readonly else self.path
        with connection(target, timeout=10, uri=self.readonly) as db:
            yield db

    def put(self, key, value):
        with self.connect() as db:
            db.execute(
                "INSERT INTO records VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM records WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def records(self, prefix):
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM records WHERE substr(id,1,?)=? ORDER BY rowid",
                    (len(prefix), prefix),
                )
            ]

    def reserve(self, operation_id, purpose):
        limits = self.get("limits", {})
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM reservations WHERE id=?", (operation_id,)).fetchone():
                return
            used = db.execute(
                "SELECT count(*) FROM reservations WHERE purpose=?", (purpose,)
            ).fetchone()[0]
            if used >= limits.get(purpose, 0):
                raise EvaluationBudgetExceeded("评测预算已用尽；未发送本次远端请求。")
            db.execute("INSERT INTO reservations VALUES (?,?,?)", (operation_id, purpose, now()))

    def usage(self):
        with self.connect() as db:
            return {
                r[0]: r[1]
                for r in db.execute("SELECT purpose,count(*) FROM reservations GROUP BY purpose")
            }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)
