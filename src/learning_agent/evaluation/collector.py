"""Read consistent runtime snapshots; never expose native reasoning or credentials."""

import json
from datetime import datetime

LOCATION = (
    "material_id",
    "name",
    "page",
    "start_line",
    "end_line",
    "start_seconds",
    "end_seconds",
    "seconds",
    "asset_path",
    "region",
    "chunk_id",
    "id",
)


def input_manifest(prepared, step):
    evidence, images = [], []
    for message in prepared.messages:
        images.extend({k: a[k] for k in LOCATION if k in a} for a in message.get("attachments", []))
        if message.get("role") != "tool":
            continue
        try:
            observation = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        if not isinstance(observation, dict):
            continue
        entry = {k: observation[k] for k in LOCATION if k in observation}
        entry["tool_call_id"] = message.get("tool_call_id")
        entry["text"] = str(observation.get("text", observation.get("excerpt", "")))[:6000]
        entry["text_truncated"] = (
            len(str(observation.get("text", observation.get("excerpt", "")))) > 6000
        )
        for field in ("hits", "segments"):
            entry[field] = [
                {k: row[k] for k in (*LOCATION, "text", "excerpt", "start", "end") if k in row}
                for row in observation.get(field, [])[:12]
            ]
        if not observation.get("error"):
            evidence.append(entry)
    snapshot = prepared.snapshot
    evidence.extend(snapshot.get("retrieval_evidence", []))
    return {
        "schema_version": 1,
        "step": step,
        "selection": snapshot.get("selection", []),
        "omissions": snapshot.get("omissions", []),
        "summary_id": snapshot.get("summary_id"),
        "memory": snapshot.get("memory", {}),
        "images": images,
        "evidence": evidence,
    }


def collect(store, rid):
    with store.connect() as db:
        db.execute("BEGIN")
        row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if row is None:
            raise LookupError("运行不存在")
        run = dict(row)
        messages = [
            dict(r)
            for r in db.execute(
                "SELECT role,kind,content FROM messages WHERE run_id=? ORDER BY rowid", (rid,)
            )
        ]
        events = [
            {**dict(r), "payload": json.loads(r["payload"])}
            for r in db.execute(
                "SELECT seq,type,payload,created_at FROM run_events WHERE run_id=? ORDER BY seq",
                (rid,),
            )
        ]
        history = [
            dict(r)
            for r in db.execute(
                "SELECT role,kind,content FROM messages WHERE conversation_id=? AND rowid<=(SELECT MAX(rowid) FROM messages WHERE run_id=?) ORDER BY rowid DESC LIMIT 8",
                (run["conversation_id"], rid),
            )
        ][::-1]
        operations = []
        for r in db.execute("SELECT * FROM run_operations WHERE run_id=? ORDER BY rowid", (rid,)):
            result = json.loads(r["result"]) if r["result"] else {}
            op = {
                k: r[k]
                for k in (
                    "id",
                    "purpose",
                    "logical_key",
                    "attempt",
                    "state",
                    "elapsed_seconds",
                    "reserved_seconds",
                    "usage_status",
                    "created_at",
                    "updated_at",
                )
            }
            op.update(
                input_ref=json.loads(r["input_ref"] or "null"),
                usage=result.get("usage"),
                observation=result.get("observation"),
            )
            operations.append(op)
        state = db.execute(
            "SELECT state_json,revision FROM learning_states WHERE conversation_id=?",
            (run["conversation_id"],),
        ).fetchone()
        draft = db.execute(
            "SELECT epoch,status,offset FROM run_drafts WHERE run_id=?", (rid,)
        ).fetchone()
    answer = next(
        (m["content"] for m in messages if m["role"] == "assistant" and m["kind"] == "answer"), None
    )
    return {
        "schema_version": 1,
        "run_id": rid,
        "conversation_id": run["conversation_id"],
        "state": run["state"],
        "revision": run["state_revision"],
        "complete": run["state"] in {"completed", "budget_exhausted", "failed", "cancelled"},
        "last_event_seq": events[-1]["seq"] if events else 0,
        "messages": messages,
        "conversation_history": history,
        "answer": answer,
        "events": events,
        "operations": operations,
        "tools": [e for e in json.loads(run["events"]) if e.get("type") == "tool"],
        "learning_state": json.loads(state[0]) if state else None,
        "learning_state_basis": "current_at_collection; isolated runner collects before next turn",
        "learning_revision": state[1] if state else None,
        "draft": dict(draft) if draft else None,
        "recovery_reason": run["recovery_reason"],
        "recovery_count": run["recovery_count"],
        "error": run["error"],
        "latency": event_latencies(events),
    }


def event_latencies(events):
    if not events:
        return {}
    start = datetime.fromisoformat(events[0]["created_at"])
    result = {}
    for event in events:
        for name, kind in (
            ("started_s", "run.started"),
            ("first_progress_s", "model.started"),
            ("first_draft_s", "answer.delta"),
        ):
            if name not in result and event["type"] == kind:
                result[name] = max(
                    0, (datetime.fromisoformat(event["created_at"]) - start).total_seconds()
                )
    if events[-1]["type"] in {"run.completed", "run.failed", "run.cancelled", "run.interrupted"}:
        result["wall_s"] = (
            datetime.fromisoformat(events[-1]["created_at"]) - start
        ).total_seconds()
    return result


def export_run(workspace, rid, *, include_content=False):
    from contextlib import contextmanager
    from pathlib import Path

    from learning_agent.database import connection
    from learning_agent.evaluation.graders import grade

    class ReadStore:
        @contextmanager
        def connect(self):
            path = Path(workspace).resolve() / "workspace.db"
            if not path.is_file():
                raise FileNotFoundError(path)
            with connection(path.as_uri() + "?mode=ro", uri=True) as db:
                yield db

    trace = collect(ReadStore(), rid)
    result = {
        "kind": "unlabelled_run_export",
        "schema_version": 1,
        "run_id": rid,
        "state": trace["state"],
        "latency": trace["latency"],
        "assessment": grade(trace, {}, {}),
        "quality": "needs_review_no_reference_labels",
    }
    if include_content:
        result["trace"] = trace
    else:
        result["operations"] = [
            {k: op[k] for k in ("id", "purpose", "state", "attempt", "usage_status", "usage")}
            for op in trace["operations"]
        ]
    return result
