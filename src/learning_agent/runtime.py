"""A supervised local scheduler. Refresh is not cancellation; restart is not replay."""

import asyncio
import fcntl
import inspect

from learning_agent.context import profile_id
from learning_agent.run_store import TERMINAL, RunStopped, RuntimeConflict, RuntimeUnavailable


class RunControl:
    def __init__(self, runtime, run_id, owner):
        self.runtime, self.store = runtime, runtime.store
        self.run_id, self.owner = run_id, owner

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
                    await asyncio.gather(task, return_exceptions=True)
                    raise
            self.check()
            return task.result()
        except BaseException:
            if cancellable:
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def tool(self, function, *args, **kwargs):
        return await self.wait(asyncio.to_thread(function, *args, **kwargs), cancellable=False)

    async def model(self, function, *, step, purpose="main_model"):
        self.check()
        op = self.store.operation(self.run_id, self.owner, purpose, step)
        self.emit("model.started", {"step": step, "purpose": purpose})
        self.store.operation(self.run_id, self.owner, purpose, step, identifier=op)
        try:
            result = await self.wait(function())
            self.store.operation(
                self.run_id,
                self.owner,
                purpose,
                step,
                identifier=op,
                result={"message": result[0], "usage": result[1]},
            )
            self.emit("model.finished", {"step": step, "purpose": purpose})
            return result
        except BaseException as error:
            with self.store.connect() as db:
                db.execute(
                    "UPDATE run_operations SET state=? WHERE id=? AND state='sending'",
                    (
                        "unknown"
                        if isinstance(error, (RunStopped, asyncio.CancelledError))
                        else "failed",
                        op,
                    ),
                )
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
            self.store.recover_runs()
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
            conversation_id, payload, profile_id(self.settings), self.settings.runtime_queue_limit
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
                    "answer": result["answer"] or "本轮已中断，请结束此轮后重新提问。",
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
                args = frozen["identity"]
                await run_turn(
                    self.store,
                    self.provider,
                    self.settings,
                    self.store.run(run_id)["conversation_id"],
                    args["content"],
                    args.get("focus"),
                    self.retriever,
                    args.get("feedback"),
                    run_id=run_id,
                    control=control,
                )
            except RunStopped as error:
                if error.interrupted:
                    self.store.interrupt_run(run_id, owner, "rt1_shutdown_requires_new_turn")
                else:
                    self.store.fail_run(run_id, owner)
            except RuntimeConflict:
                self.store.interrupt_run(run_id, owner, "context_changed")
            except asyncio.CancelledError:
                self.store.interrupt_run(run_id, owner, "runtime_cancelled")
                raise
            except Exception:  # noqa: BLE001 - isolate unexpected failures in supervised runs
                self.store.fail_run(run_id, owner)
