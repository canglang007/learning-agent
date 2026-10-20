"""A supervised local scheduler. Refresh is not cancellation; restart is not replay."""

import asyncio
import copy
import fcntl
import inspect
import json
import time

from learning_agent.context import profile_id
from learning_agent.execution_store import UnknownRequest, execution_profile
from learning_agent.provider import ModelError
from learning_agent.run_store import TERMINAL, RunStopped, RuntimeConflict, RuntimeUnavailable


class RunControl:
    def __init__(self, runtime, run_id, owner):
        self.runtime, self.store = runtime, runtime.store
        self.run_id, self.owner = run_id, owner
        saved = self.store.checkpoint(run_id)
        self.saved = saved["payload"] if saved else None
        self.checkpoint_revision = saved["revision"] if saved else 0
        self.observed = copy.deepcopy(self.saved.get("observed", {})) if self.saved else {}
        self.active_tool = None
        self.query_key = None
        self.query_tasks = set()

    def check(self):
        row = self.store.run(self.run_id)
        if row["owner_token"] != self.owner or row["state"] not in {"running", "cancelling"}:
            raise RunStopped(interrupted=True)
        if row["cancel_requested"]:
            raise RunStopped()
        if self.runtime.stopping:
            raise RunStopped(interrupted=True)

    def emit(self, event, payload=None):
        self.check()
        self.store.progress(self.run_id, self.owner, event, payload)

    async def wait(self, awaitable, *, cancellable=True):
        try:
            self.check()
        except RunStopped:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise
        task = asyncio.create_task(awaitable)
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                try:
                    self.check()
                except RunStopped:
                    if cancellable:
                        task.cancel()
                    else:
                        for query in self.query_tasks:
                            if query is not asyncio.current_task():
                                query.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise
            self.check()
            return task.result()
        except BaseException:
            if cancellable:
                task.cancel()
            else:
                for query in self.query_tasks:
                    if query is not asyncio.current_task():
                        query.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    def checkpoint(self, phase, payload, *, event=None, finish_tool=False):
        self.check()
        payload["observed"] = copy.deepcopy(self.observed)
        operation, elapsed = None, 0
        if finish_tool and self.active_tool:
            operation, started = self.active_tool
            elapsed = time.monotonic() - started
        self.checkpoint_revision = self.store.save_checkpoint(
            self.run_id,
            self.owner,
            self.checkpoint_revision,
            phase,
            payload,
            operation=operation,
            result={"observation": payload.get("last_observation")},
            elapsed=elapsed,
            event=event,
        )
        self.saved = copy.deepcopy(payload)
        if finish_tool:
            self.active_tool = None

    def validate(self):
        if self.store.run_input(self.run_id).get("execution"):
            self.store.validate_execution(self.run_id, self.runtime.settings)

    def final_guard(self, db):
        self.store.validate_execution(self.run_id, self.runtime.settings, db=db)
        self.verify_observed()
        checkpoint = db.execute(
            "SELECT revision,phase FROM run_checkpoints WHERE run_id=?", (self.run_id,)
        ).fetchone()
        if (
            not checkpoint
            or checkpoint["revision"] != self.checkpoint_revision
            or checkpoint["phase"] != "ready_to_commit"
        ):
            raise RuntimeConflict("最终交付检查点已改变")

    def elapsed(self):
        # Across restart, unknown attempts consume their conservative reserved duration.
        with self.store.connect() as db:
            return db.execute(
                "SELECT COALESCE(SUM(CASE WHEN state='unknown' THEN reserved_seconds WHEN elapsed_seconds IS NOT NULL THEN elapsed_seconds ELSE 0 END),0) FROM run_operations WHERE run_id=? AND purpose!='query_embedding'",
                (self.run_id,),
            ).fetchone()[0]

    def observe(self, name, args):
        self.validate()
        self.verify_observed()
        versions = self.store.material_versions(self.store.run_input(self.run_id)["space_id"])
        mid = args.get("material_id") if isinstance(args, dict) else None
        if mid and mid in versions:
            self.observed[mid] = versions[mid]
        elif name in {"list_materials", "search_materials", "read_context"}:
            self.observed["*"] = versions
        return versions

    def verify_observed(self):
        current = self.store.material_versions(self.store.run_input(self.run_id)["space_id"])
        if any(current.get(k) != v for k, v in self.observed.items() if k != "*") or (
            "*" in self.observed and current != self.observed["*"]
        ):
            raise RuntimeConflict("已查阅资料版本改变")

    def bind_assets(self, prepared):
        from pathlib import Path

        versions = self.store.material_versions(self.store.run_input(self.run_id)["space_id"])
        self.verify_observed()
        for message in prepared.messages:
            for image in message.get("attachments", []):
                parts = Path(image["asset_path"]).parts
                if len(parts) > 2 and parts[0] == "materials" and parts[1] in versions:
                    self.observed[parts[1]] = versions[parts[1]]

    async def restore_assets(self, tools, groups, prepared, prior_turns, budget, pending=()):
        from learning_agent.provider import asset_info
        from learning_agent.render_process import render_tool

        provenance, recorded_versions = {}, {}
        source_events = list((self.saved or {}).get("events", []))
        for turn in prior_turns:
            if turn["run_id"]:
                source_events.extend(self.store.run(turn["run_id"])["events"])
        for event in source_events:
            observation = event.get("observation", {})
            for image in (
                [observation["image"]] if "image" in observation else []
            ) + observation.get("images", []):
                recorded_versions[image["asset_path"]] = event.get("material_revision")
        batches = [g.dump() for g in groups]
        for turn in prior_turns:
            batches.extend(turn["groups"])
        for group in batches:
            for message in group["messages"]:
                if message["role"] != "tool":
                    continue
                try:
                    observation = json.loads(message["content"])
                except (ValueError, TypeError):
                    continue
                for image in (
                    [observation["image"]] if "image" in observation else []
                ) + observation.get("images", []):
                    provenance[image["asset_path"]] = (observation, image)
        # Rebuild only this run's images and the exact previously prepared input.
        selected = {
            image["asset_path"]: image
            for g in groups
            for m in g.messages
            for image in m.get("attachments", [])
        }
        selected.update({image["asset_path"]: image for image, _label in pending})
        if prepared:
            selected.update(
                {
                    image["asset_path"]: image
                    for m in prepared.messages
                    for image in m.get("attachments", [])
                }
            )
        restored = 0
        for path, image in selected.items():
            try:
                asset_info(self.store.root, image)
                continue
            except ModelError:
                pass
            if restored >= budget or path not in provenance:
                raise RuntimeConflict("图像缺失且没有可用的恢复来源或预算")
            observation, description = provenance[path]
            mid = observation["material_id"]
            revision = recorded_versions.get(path)
            if (
                revision is not None
                and self.store.material(tools.space_id, mid)["revision"] != revision
            ):
                raise RuntimeConflict("缺失图像的原资料版本已改变")
            self.observe("inspect_page", {"material_id": mid})
            if "seconds" in description:
                name, args = (
                    "_restore_frame",
                    {"material_id": mid, "seconds": description["seconds"]},
                )
            else:
                name, args = (
                    "_restore_page",
                    {
                        "material_id": mid,
                        "page": observation["page"],
                        "region": observation.get("region"),
                    },
                )
            key = f"asset:{self.store.run(self.run_id)['recovery_count']}:{path}"
            op, cached = self.store.prepare_operation(
                self.run_id,
                self.owner,
                "tool",
                0,
                key,
                args,
                self.runtime.settings.timeout_seconds,
                images=1,
            )
            if cached is not None:
                raise RuntimeConflict("已恢复图像再次丢失，请重新提问")
            self.store.send_operation(self.run_id, self.owner, op["id"])
            started = time.monotonic()
            async with asyncio.timeout(max(0.01, op["reserved_seconds"])):
                result = await self.wait(render_tool(self, tools, name, args))
            if result["image"]["asset_path"] != path:
                raise RuntimeConflict("恢复图像位置改变")
            self.store.complete_operation(
                self.run_id,
                self.owner,
                op["id"],
                result,
                time.monotonic() - started,
                event=("asset.restored", {}),
            )
            restored += 1
        return restored

    def begin_tool(self, step, index, name, args):
        self.check()
        self.observe(name, args)
        self.query_key = f"embedding:{step}:{index}"
        self.active_step, self.active_index = step, index
        requested = (
            1
            if name == "inspect_page"
            else args.get("frame_count", 3)
            if name == "inspect_media"
            and isinstance(args, dict)
            and args.get("mode", "joint") == "joint"
            else 0
        )
        if not isinstance(requested, int) or requested < 0:
            requested = 0
        images = min(
            requested,
            max(0, self.runtime.settings.max_images_per_turn - self.store.image_usage(self.run_id)),
        )
        op, _ = self.store.prepare_operation(
            self.run_id,
            self.owner,
            "tool",
            step,
            f"tool:{step}:{index}",
            {"name": name, "arguments": args, "observed": copy.deepcopy(self.observed)},
            self.runtime.settings.timeout_seconds,
            images=images,
        )
        self.store.send_operation(self.run_id, self.owner, op["id"])
        self.active_tool = (op["id"], time.monotonic())
        return op["reserved_seconds"]

    async def tool(self, function, *args, **kwargs):
        from learning_agent.render_process import render_tool

        # Only rendering leaves the API process; ordinary reads remain inexpensive threads.
        op = (
            self.store.operation_at(self.run_id, f"tool:{self.active_step}:{self.active_index}")
            if self.active_tool
            else None
        )
        async with asyncio.timeout(
            max(0.01, op["reserved_seconds"]) if op else self.runtime.settings.timeout_seconds
        ):
            if args and args[0] in {"inspect_page", "inspect_media"}:
                result = await self.wait(render_tool(self, function.__self__, *args, **kwargs))
            else:
                result = await self.wait(
                    asyncio.to_thread(function, *args, **kwargs), cancellable=False
                )
        self.validate()
        self.verify_observed()
        return result

    async def query_embedding(self, embedder, texts):
        from learning_agent.retrieval import EmbeddingError

        self.check()
        key = self.query_key
        input_ref = {
            "texts": texts,
            "model": self.runtime.settings.embedding_model,
            "service": execution_profile(self.runtime.settings)["embedding_service"],
        }
        try:
            op, cached = self.store.prepare_operation(
                self.run_id, self.owner, "query_embedding", 0, key, input_ref, 0
            )
        except UnknownRequest:
            raise EmbeddingError("remote_result_unknown:bm25_fallback") from None
        except ModelError:
            previous = self.store.operation_at(self.run_id, key)
            raise EmbeddingError(
                "embedding_previous_" + previous["state"]
                if previous
                else "embedding_budget_exhausted"
            ) from None
        if cached is not None:
            return cached["vectors"]
        self.store.send_operation(self.run_id, self.owner, op["id"])
        started = time.monotonic()
        current = asyncio.current_task()
        self.query_tasks.add(current)
        try:
            if hasattr(embedder, "embed_async"):
                result = await self.wait(embedder.embed_async(texts))
            else:
                result = await self.wait(
                    asyncio.to_thread(embedder.embed, texts), cancellable=False
                )
            self.store.complete_operation(
                self.run_id, self.owner, op["id"], {"vectors": result}, time.monotonic() - started
            )
            return result
        except BaseException as error:
            self.store.operation_failed(
                self.run_id,
                self.owner,
                op["id"],
                getattr(error, "outcome", "unknown"),
                elapsed=time.monotonic() - started,
            )
            raise
        finally:
            self.query_tasks.discard(current)

    async def model(self, function, *, step, purpose="main_model", input_ref=None, stream=None):
        self.check()
        self.validate()
        self.verify_observed()
        key = f"{purpose}:{step}"
        try:
            op, cached = self.store.prepare_operation(
                self.run_id,
                self.owner,
                purpose,
                step,
                key,
                input_ref or {"step": step, "purpose": purpose},
                self.runtime.settings.summary_timeout_seconds
                if purpose == "summary"
                else self.runtime.settings.timeout_seconds,
            )
        except UnknownRequest:
            if purpose == "main_model":
                raise
            raise ModelError("摘要请求结果未知，本轮沿用裁减，不自动重发") from None
        if cached is not None:
            return cached["message"], cached["usage"]
        if purpose == "main_model":
            self.store.revoke_draft(self.run_id, self.owner, "new_model_request")
        if self.runtime.settings.evaluation_budget_path:
            from learning_agent.evaluation.ledger import EvaluationBudgetExceeded, Ledger

            try:
                Ledger(self.runtime.settings.evaluation_budget_path).reserve(op["id"], purpose)
            except EvaluationBudgetExceeded:
                self.emit("evaluation.budget_exhausted", {"purpose": purpose})
                raise
        self.emit("model.started", {"step": step, "purpose": purpose})
        self.store.send_operation(self.run_id, self.owner, op["id"])
        started = time.monotonic()
        try:
            async with asyncio.timeout(max(0.01, op["reserved_seconds"])):
                if stream is not None and purpose == "main_model":
                    from learning_agent.drafts import DraftSink

                    result = await self.wait(stream(DraftSink(self, op["id"])))
                else:
                    result = await self.wait(function())
            self.store.complete_operation(
                self.run_id,
                self.owner,
                op["id"],
                {"message": result[0], "usage": result[1]},
                time.monotonic() - started,
                event=("model.finished", {"step": step, "purpose": purpose}),
            )
            return result
        except BaseException as error:
            if purpose == "main_model":
                self.store.revoke_draft(self.run_id, self.owner, "model_not_completed")
            state = (
                "unknown"
                if isinstance(error, (RunStopped, asyncio.CancelledError, TimeoutError))
                or getattr(error, "outcome", None) == "unknown"
                else "returned_invalid"
                if getattr(error, "outcome", None) == "returned_invalid"
                else "failed"
            )
            self.store.operation_failed(
                self.run_id,
                self.owner,
                op["id"],
                state,
                elapsed=time.monotonic() - started,
                usage=getattr(error, "usage", None),
            )
            if state == "unknown" and not isinstance(error, (RunStopped, asyncio.CancelledError)):
                if purpose == "main_model":
                    raise UnknownRequest(purpose) from None
                raise ModelError("摘要请求结果未知，沿用裁减") from None
            raise


class RuntimeService:
    def __init__(self, store, provider, settings, retriever):
        self.store, self.provider, self.settings, self.retriever = (
            store,
            provider,
            settings,
            retriever,
        )
        self.stopping, self.task, self.lock = False, None, None
        self.wake = asyncio.Event()

    async def start(self):
        self.stopping = False
        self.wake = asyncio.Event()
        self.lock = (self.store.root / ".runtime.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError(
                "该数据目录已有问答调度器，请使用一个API进程（--workers 1）"
            ) from None
        try:
            self.store.recover_runs(self.settings)
            self.task = asyncio.create_task(self.run(), name="learning-runtime")
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    async def close(self):
        self.stopping = True
        self.wake.set()
        try:
            if self.task:
                await self.task
        finally:
            if self.lock:
                self.lock.close()
                self.lock = None

    def submit(self, conversation_id, payload):
        if self.task is None or self.task.done() or self.stopping:
            raise RuntimeUnavailable("问答调度器尚未启动或正在关闭，请检查API服务")
        run_id = self.store.submit_run(
            conversation_id,
            payload,
            profile_id(self.settings),
            self.settings.runtime_queue_limit,
            execution=execution_profile(self.settings),
        )
        self.wake.set()
        return run_id

    async def wait_result(self, run_id):
        while True:
            result = self.store.public_run(run_id)
            if result["state"] in TERMINAL or result["state"] == "interrupted":
                return {
                    "run_id": run_id,
                    "state": result["state"],
                    "answer": result["answer"]
                    or "本轮已中断，请查看任务的继续、明确重试或结束入口。",
                    "context_limited": any(
                        s.get("omissions") for s in self.store.run_context(run_id)
                    ),
                }
            await asyncio.sleep(0.05)

    async def run(self):
        from learning_agent.agent import run_turn

        while not self.stopping:
            claimed = self.store.claim_run()
            if not claimed:
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=0.25)
                except TimeoutError:
                    pass
                continue
            run_id, owner = claimed
            control = RunControl(self, run_id, owner)
            try:
                frozen = self.store.run_input(run_id)
                from learning_agent.provider import OpenAIProvider

                if (
                    isinstance(self.provider, OpenAIProvider)
                    and not self.settings.api_key.get_secret_value()
                ):
                    raise RuntimeConflict("模型未配置")
                state = self.store.current_learning_state(self.store.run(run_id)["conversation_id"])
                if (
                    frozen["profile"] != profile_id(self.settings)
                    or state["revision"] != frozen["learning"]["revision"]
                ):
                    raise RuntimeConflict("提交后配置或学习状态改变")
                if frozen.get("execution"):
                    self.store.validate_execution(run_id, self.settings)
                run_retriever = copy.copy(self.retriever)
                loop = asyncio.get_running_loop()
                run_retriever.query_embed = (
                    lambda texts, control=control, embedder=run_retriever.embedder, loop=loop: (
                        asyncio.run_coroutine_threadsafe(
                            control.query_embedding(embedder, texts), loop
                        ).result()
                    )
                )
                args = frozen["identity"]
                await run_turn(
                    self.store,
                    self.provider,
                    self.settings,
                    self.store.run(run_id)["conversation_id"],
                    args["content"],
                    args.get("focus"),
                    run_retriever,
                    args.get("feedback"),
                    run_id=run_id,
                    control=control,
                )
            except UnknownRequest:
                self.store.interrupt_run(run_id, owner, "remote_result_unknown")
            except RunStopped as error:
                if error.interrupted:
                    self.store.interrupt_run(run_id, owner, "process_interrupted")
                else:
                    self.store.fail_run(run_id, owner)
            except RuntimeConflict:
                self.store.interrupt_run(run_id, owner, "context_changed")
            except asyncio.CancelledError:
                self.store.interrupt_run(run_id, owner, "runtime_cancelled")
                raise
            except Exception:  # noqa: BLE001 - isolate unexpected failures in supervised runs
                self.store.fail_run(run_id, owner)
