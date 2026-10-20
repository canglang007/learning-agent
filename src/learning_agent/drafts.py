"""Durable, coalesced provisional text. Never part of chat or model context."""

import time

from learning_agent.run_store import RunStopped, now


class DraftStore:
    def init_drafts(self):
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS run_drafts (
                run_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, operation_id TEXT NOT NULL,
                status TEXT NOT NULL, text TEXT NOT NULL, offset INTEGER NOT NULL,
                updated_at TEXT NOT NULL)""")

    def draft_snapshot(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM run_drafts WHERE run_id=?", (rid,)).fetchone()
        return self.public_draft(row)

    @staticmethod
    def public_draft(row):
        return (
            {k: row[k] for k in ("epoch", "status", "text", "offset", "updated_at")}
            if row
            else None
        )

    def revoke_draft_in(self, db, rid, reason):
        row = db.execute("SELECT epoch,status FROM run_drafts WHERE run_id=?", (rid,)).fetchone()
        if row and row["status"] in {"generating", "validating"}:
            db.execute(
                "UPDATE run_drafts SET status='revoked',updated_at=? WHERE run_id=?", (now(), rid)
            )
            self.event_in(db, rid, "draft.revoked", {"epoch": row["epoch"], "reason": reason})

    def revoke_draft(self, rid, owner, reason):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner_token FROM runs WHERE id=?", (rid,)).fetchone()
            if row and row["owner_token"] == owner:
                self.revoke_draft_in(db, rid, reason)

    def write_draft(self, rid, owner, operation, text, *, final=False):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.execution_owner(db, rid, owner)
            op = db.execute(
                "SELECT owner_token,state FROM run_operations WHERE id=? AND run_id=?",
                (operation, rid),
            ).fetchone()
            if not op or op["owner_token"] != owner or op["state"] != "sending":
                raise RunStopped(interrupted=True)
            row = db.execute("SELECT * FROM run_drafts WHERE run_id=?", (rid,)).fetchone()
            if row and row["operation_id"] == operation:
                if row["status"] == "revoked":
                    return
                if not text.startswith(row["text"]):
                    self.revoke_draft_in(db, rid, "projection_changed")
                    return
                epoch, start = row["epoch"], row["offset"]
            else:
                self.revoke_draft_in(db, rid, "new_model_request")
                epoch, start = (row["epoch"] + 1 if row else 1), 0
            status = "validating" if final else "generating"
            db.execute(
                "INSERT INTO run_drafts VALUES (?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET epoch=excluded.epoch,operation_id=excluded.operation_id,status=excluded.status,text=excluded.text,offset=excluded.offset,updated_at=excluded.updated_at",
                (rid, epoch, operation, status, text, len(text), now()),
            )
            if len(text) > start:
                self.event_in(
                    db, rid, "answer.delta", {"epoch": epoch, "offset": start, "text": text[start:]}
                )
            if final:
                self.event_in(db, rid, "draft.validating", {"epoch": epoch})


class DraftSink:
    def __init__(self, control, operation):
        self.control, self.operation = control, operation
        self.text, self.saved, self.last, self.mode = "", "", 0, None
        self.revoked = False

    def __call__(self, text, mode, final=False):
        self.control.check()
        if mode != self.mode:
            self.control.emit("stream.mode", {"mode": mode})
            self.mode = mode
        if self.revoked:
            return
        if not text.startswith(self.text):
            self.control.store.revoke_draft(
                self.control.run_id, self.control.owner, "projection_changed"
            )
            self.revoked = True
            return
        self.text = text
        if not text:
            return
        elapsed = time.monotonic() - self.last
        # First decoded text is immediate; later fragments coalesce by time or size.
        if (
            final
            or not self.saved
            or len(text) - len(self.saved) >= 128
            or (text != self.saved and elapsed >= 0.2)
        ):
            self.control.store.write_draft(
                self.control.run_id, self.control.owner, self.operation, text, final=final
            )
            self.saved, self.last = text, time.monotonic()
