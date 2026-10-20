"""Versioned execution checkpoints and operation attempts, separate from chat history."""

import json
from uuid import uuid4

from learning_agent.context import profile_id
from learning_agent.provider import ModelError
from learning_agent.run_store import RunStopped, RuntimeConflict, now

EXECUTION_VERSION = "learning-loop-v2"


def execution_profile(settings):
    # Effective policy is frozen; credentials and credential-bearing URLs are never stored.
    fields = (
        "timeout_seconds",
        "stream_answer",
        "max_steps",
        "enable_vision",
        "max_images_per_turn",
        "context_budget_tokens",
        "context_output_reserve_tokens",
        "context_headroom_tokens",
        "max_context_images",
        "context_image_allowance",
        "context_max_request_bytes",
        "output_token_field",
        "enable_context_summary",
        "summary_input_budget_tokens",
        "summary_timeout_seconds",
        "summary_min_new_turns",
        "summary_keep_recent_turns",
        "retrieval_mode",
        "embedding_provider",
        "embedding_model",
        "embedding_timeout_seconds",
    )
    embedding = settings.model_copy(update={"base_url": settings.embedding_base_url})
    return {
        "version": EXECUTION_VERSION,
        "model": profile_id(settings),
        "embedding_service": profile_id(embedding)["service"],
        "policy": {k: getattr(settings, k) for k in fields},
    }


class UnknownRequest(Exception):
    pass


class ExecutionStore:
    def init_execution(self):
        with self.connect() as db:
            for table, columns in {
                "runs": {
                    "recovery_count": "INTEGER NOT NULL DEFAULT 0",
                    "remaining_seconds": "REAL",
                },
                "run_checkpoints": {
                    "schema_version": "INTEGER NOT NULL DEFAULT 1",
                    "revision": "INTEGER NOT NULL DEFAULT 0",
                },
                "run_operations": {
                    "logical_key": "TEXT",
                    "input_ref": "TEXT",
                    "attempt": "INTEGER NOT NULL DEFAULT 1",
                    "owner_token": "TEXT",
                    "reserved_seconds": "REAL NOT NULL DEFAULT 0",
                    "elapsed_seconds": "REAL",
                    "acknowledged": "INTEGER NOT NULL DEFAULT 0",
                    "usage_status": "TEXT NOT NULL DEFAULT 'unknown'",
                    "image_reserved": "INTEGER NOT NULL DEFAULT 0",
                },
            }.items():
                existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                for name, declaration in columns.items():
                    if name not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
            db.execute(
                "CREATE TABLE IF NOT EXISTS run_actions (run_id TEXT, action_id TEXT, arguments TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY(run_id,action_id))"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS operation_cursor ON run_operations(run_id,logical_key,attempt)"
            )

    @staticmethod
    def execution_owner(db, rid, owner):
        row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if not row or row["owner_token"] != owner:
            raise RunStopped(interrupted=True)
        if row["cancel_requested"]:
            raise RunStopped()
        if row["state"] != "running":
            raise RunStopped(interrupted=True)
        return row

    def checkpoint(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM run_checkpoints WHERE run_id=?", (rid,)).fetchone()
        if not row or row["schema_version"] != 2:
            return None
        return {**dict(row), "payload": json.loads(row["payload"])}

    def save_checkpoint(
        self,
        rid,
        owner,
        revision,
        phase,
        payload,
        *,
        operation=None,
        result=None,
        elapsed=0,
        event=None,
    ):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.execution_owner(db, rid, owner)
            old = db.execute(
                "SELECT revision,schema_version FROM run_checkpoints WHERE run_id=?", (rid,)
            ).fetchone()
            if (old["revision"] if old and old["schema_version"] == 2 else 0) != revision:
                raise RuntimeConflict("执行检查点已改变")
            if operation:
                self._complete_operation(db, rid, owner, operation, result, elapsed)
            db.execute(
                "INSERT INTO run_checkpoints(run_id,phase,payload,updated_at,schema_version,revision) VALUES (?,?,?,?,2,?) "
                "ON CONFLICT(run_id) DO UPDATE SET phase=excluded.phase,payload=excluded.payload,updated_at=excluded.updated_at,schema_version=2,revision=excluded.revision",
                (rid, phase, json.dumps(payload, ensure_ascii=False), now(), revision + 1),
            )
            db.execute(
                "UPDATE runs SET phase=?,events=?,updated_at=? WHERE id=?",
                (phase, json.dumps(payload.get("events", []), ensure_ascii=False), now(), rid),
            )
            if event:
                self.event_in(db, rid, event[0], event[1])
        return revision + 1

    def operation_at(self, rid, key):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM run_operations WHERE run_id=? AND logical_key=? ORDER BY attempt DESC LIMIT 1",
                (rid, key),
            ).fetchone()
        return (
            {
                **dict(row),
                "input_ref": json.loads(row["input_ref"]),
                "result": json.loads(row["result"]) if row["result"] else None,
            }
            if row
            else None
        )

    def prepare_operation(self, rid, owner, purpose, step, key, input_ref, timeout, *, images=0):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.execution_owner(db, rid, owner)
            previous = db.execute(
                "SELECT * FROM run_operations WHERE run_id=? AND logical_key=? ORDER BY attempt DESC LIMIT 1",
                (rid, key),
            ).fetchone()
            if previous:
                if json.loads(previous["input_ref"]) != input_ref:
                    raise RuntimeConflict("恢复操作的输入或资料范围改变")
                if previous["state"] == "returned":
                    return dict(previous), json.loads(previous["result"])
                if previous["state"] == "prepared":
                    db.execute(
                        "UPDATE run_operations SET owner_token=? WHERE id=?",
                        (owner, previous["id"]),
                    )
                    return {**dict(previous), "owner_token": owner}, None
                if (
                    previous["state"] in {"sending", "unknown"}
                    and purpose != "tool"
                    and not previous["acknowledged"]
                ):
                    raise UnknownRequest(purpose)
                if previous["state"] in {"failed", "returned_invalid"}:
                    raise ModelError("先前操作已失败，本轮不自动重发")
            frozen = self.run_input(rid)
            policy = (frozen.get("execution") or {}).get("policy", {})
            limit = (
                policy.get("max_steps", 8)
                if purpose == "main_model"
                else 12
                if purpose in {"tool", "query_embedding"}
                else 1
            )
            used = db.execute(
                "SELECT count(*) FROM run_operations WHERE run_id=? AND purpose=?", (rid, purpose)
            ).fetchone()[0]
            if used >= limit:
                raise ModelError("本轮持久执行预算已用完，请结束后重新提问。")
            remaining = row["remaining_seconds"]
            if remaining is None:
                remaining = policy.get("timeout_seconds", 90) * 2
            if remaining <= 0 and purpose != "query_embedding":
                raise ModelError("本轮有效执行时间预算已用完，重启不会重置预算。")
            reserved = min(timeout, remaining)
            image_limit = policy.get("max_images_per_turn", 3)
            image_used = db.execute(
                "SELECT COALESCE(SUM(image_reserved),0) FROM run_operations WHERE run_id=?", (rid,)
            ).fetchone()[0]
            if image_used + images > image_limit:
                raise ModelError("本轮持久图像预算已用完，恢复不会重置额度。")
            identifier, stamp = str(uuid4()), now()
            db.execute(
                "UPDATE runs SET remaining_seconds=? WHERE id=?", (remaining - reserved, rid)
            )
            db.execute(
                "INSERT INTO run_operations(id,run_id,purpose,step,state,created_at,updated_at,logical_key,input_ref,attempt,owner_token,reserved_seconds,image_reserved) VALUES (?,?,?,?,'prepared',?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    rid,
                    purpose,
                    step,
                    stamp,
                    stamp,
                    key,
                    json.dumps(input_ref, ensure_ascii=False),
                    (previous["attempt"] + 1) if previous else 1,
                    owner,
                    reserved,
                    images,
                ),
            )
            return {"id": identifier, "reserved_seconds": reserved, "state": "prepared"}, None

    def send_operation(self, rid, owner, op):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.execution_owner(db, rid, owner)
            changed = db.execute(
                "UPDATE run_operations SET state='sending',owner_token=?,updated_at=? WHERE id=? AND run_id=? AND state='prepared'",
                (owner, now(), op, rid),
            ).rowcount
            if changed != 1:
                raise RuntimeConflict("操作发送边界已改变")

    def _complete_operation(self, db, rid, owner, op, result, elapsed):
        self.execution_owner(db, rid, owner)
        operation = db.execute(
            "SELECT * FROM run_operations WHERE id=? AND run_id=?", (op, rid)
        ).fetchone()
        if not operation or operation["owner_token"] != owner:
            raise RunStopped(interrupted=True)
        if operation["state"] == "returned":
            return
        if operation["state"] != "sending":
            raise RuntimeConflict("操作完成边界已改变")
        observation = result.get("observation") or result
        returned_images = int("image" in observation) + len(observation.get("images", []))
        if returned_images > operation["image_reserved"]:
            raise RuntimeConflict("图像结果超出已预留预算")
        db.execute(
            "UPDATE run_operations SET state='returned',result=?,elapsed_seconds=?,usage_status=?,updated_at=? WHERE id=?",
            (
                json.dumps(result, ensure_ascii=False),
                elapsed,
                "known" if result.get("usage") else "unavailable",
                now(),
                op,
            ),
        )
        refund = (
            max(0, operation["reserved_seconds"] - elapsed) if operation["reserved_seconds"] else 0
        )
        db.execute(
            "UPDATE runs SET remaining_seconds=remaining_seconds+? WHERE id=?", (refund, rid)
        )
        db.execute("UPDATE run_operations SET image_reserved=? WHERE id=?", (returned_images, op))

    def image_usage(self, rid):
        with self.connect() as db:
            return db.execute(
                "SELECT COALESCE(SUM(image_reserved),0) FROM run_operations WHERE run_id=?", (rid,)
            ).fetchone()[0]

    def complete_operation(self, rid, owner, op, result, elapsed, event=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._complete_operation(db, rid, owner, op, result, elapsed)
            if event:
                self.event_in(db, rid, event[0], event[1])

    def operation_failed(self, rid, owner, op, state, *, elapsed=None, usage=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner_token FROM runs WHERE id=?", (rid,)).fetchone()
            if row and row[0] == owner:
                operation = db.execute(
                    "SELECT reserved_seconds,state FROM run_operations WHERE id=?", (op,)
                ).fetchone()
                changed = db.execute(
                    "UPDATE run_operations SET state=?,updated_at=? WHERE id=? AND run_id=? AND owner_token=? AND state='sending'",
                    (state, now(), op, rid, owner),
                ).rowcount
                if changed and state != "unknown" and elapsed is not None:
                    refund = max(0, operation["reserved_seconds"] - elapsed)
                    db.execute(
                        "UPDATE runs SET remaining_seconds=remaining_seconds+? WHERE id=?",
                        (refund, rid),
                    )
                if changed:
                    db.execute(
                        "UPDATE run_operations SET elapsed_seconds=?,usage_status=?,result=? WHERE id=?",
                        (
                            elapsed,
                            "known"
                            if isinstance(usage, dict) and usage
                            else "unknown"
                            if state == "unknown"
                            else "unavailable",
                            json.dumps({"usage": usage}) if isinstance(usage, dict) else None,
                            op,
                        ),
                    )

    def operation_count(self, rid, purpose):
        with self.connect() as db:
            return db.execute(
                "SELECT count(*) FROM run_operations WHERE run_id=? AND purpose=?", (rid, purpose)
            ).fetchone()[0]

    def material_versions(self, space):
        with self.connect() as db:
            return {
                r["id"]: r["revision"]
                for r in db.execute(
                    "SELECT id,revision FROM materials WHERE space_id=? ORDER BY id", (space,)
                )
            }

    def validate_execution(self, rid, settings, *, db=None):
        frozen = self.run_input(rid)
        if frozen.get("execution") != execution_profile(settings):
            raise RuntimeConflict("执行配置或策略版本改变")
        cid = self.run(rid)["conversation_id"]
        if db is None:
            state = self.current_learning_state(cid)
        else:
            state = dict(
                db.execute(
                    "SELECT revision FROM learning_states WHERE conversation_id=?", (cid,)
                ).fetchone()
            )
        if (
            state["revision"] != frozen["learning"]["revision"]
            or self.context_settings(cid) != frozen["context_settings"]
        ):
            raise RuntimeConflict("学习状态或接续设置改变")
        cp = self.checkpoint(rid)
        if cp:
            observed = cp["payload"].get("observed", {})
            current = self.material_versions(frozen["space_id"])
            if any(current.get(k) != v for k, v in observed.items() if k != "*") or (
                "*" in observed and current != observed["*"]
            ):
                raise RuntimeConflict("已查阅资料版本或搜索范围改变")

    def recovery_options(self, rid):
        row = self.run(rid)
        frozen = self.run_input(rid) if row["client_request_id"] else {}
        cp = self.checkpoint(rid)
        if (
            not frozen.get("execution")
            or row["recovery_count"] >= 3
            or row["remaining_seconds"] is not None
            and row["remaining_seconds"] <= 0
        ):
            return None
        if row["recovery_reason"] in {"context_changed", "assets_unavailable", "recovery_limit"}:
            return None
        if not cp and self.operation_count(rid, "main_model"):
            return None
        with self.connect() as db:
            unknown = db.execute(
                "SELECT id FROM run_operations WHERE run_id=? AND purpose='main_model' AND state IN ('sending','unknown') AND acknowledged=0",
                (rid,),
            ).fetchone()
        if unknown:
            if (
                self.operation_count(rid, "main_model")
                >= frozen["execution"]["policy"]["max_steps"]
            ):
                return None
            return "retry_model"
        return "resume"

    def resume_run(self, rid, settings, action_id, expected_revision, acknowledge=False):
        arguments = {
            "expected_revision": expected_revision,
            "acknowledge_unknown_request": acknowledge,
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM run_actions WHERE run_id=? AND action_id=?", (rid, action_id)
            ).fetchone()
            if prior:
                if json.loads(prior["arguments"]) != arguments:
                    raise RuntimeConflict("同一操作ID不能用于不同恢复请求")
                return json.loads(prior["result"])
            row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
            if not row:
                raise LookupError("问答任务不存在")
            if row["state"] != "interrupted" or row["state_revision"] != expected_revision:
                raise RuntimeConflict("任务状态已改变，请重新查询")
            self.validate_execution(rid, settings, db=db)
            action = self.recovery_options(rid)
            if action is None:
                raise RuntimeConflict("本轮没有可继续的执行路径，请结束后重新提问")
            if action == "retry_model" and not acknowledge:
                raise RuntimeConflict("先前模型请求结果未知，重试可能再次计费，请明确确认")
            db.execute(
                "UPDATE run_operations SET acknowledged=1 WHERE run_id=? AND purpose='main_model' AND state='unknown'",
                (rid,),
            )
            db.execute(
                "UPDATE runs SET state='queued',owner_token=NULL,recovery_count=recovery_count+1,recovery_reason=NULL,state_revision=state_revision+1,phase='queued',updated_at=? WHERE id=?",
                (now(), rid),
            )
            self.event_in(db, rid, "run.resumed", {"action": action})
            result = {"run_id": rid, "state": "queued", "revision": row["state_revision"] + 1}
            db.execute(
                "INSERT INTO run_actions VALUES (?,?,?,?)",
                (rid, action_id, json.dumps(arguments), json.dumps(result)),
            )
        return result
