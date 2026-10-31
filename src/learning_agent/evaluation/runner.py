import asyncio
import json
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.context import profile_id
from learning_agent.evaluation.collector import collect
from learning_agent.evaluation.ledger import Ledger, save_json
from learning_agent.retrieval import INDEX_VERSION, Retriever
from learning_agent.run_store import now
from learning_agent.store import Store

ALLOWED = {
    "model",
    "protocol",
    "route_policy",
    "retrieval_mode",
    "embedding_provider",
    "embedding_model",
    "max_steps",
    "context_budget_tokens",
    "context_headroom_tokens",
    "context_output_reserve_tokens",
    "max_context_images",
    "context_image_allowance",
    "enable_context_summary",
    "summary_input_budget_tokens",
    "summary_min_new_turns",
    "summary_keep_recent_turns",
    "stream_answer",
}


def configuration(base, changes, **internal):
    if set(changes) - ALLOWED:
        raise ValueError("评测配置包含不支持的字段：" + ",".join(sorted(set(changes) - ALLOWED)))
    return Settings(_env_file=None, **{**base.model_dump(), **changes, **internal})


def plan(suite, variants, cases=None, split="dev", repeats=1):
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", name) for name in variants):
        raise ValueError("配置名称只能含字母数字下划线和连字符")
    selected = [
        c
        for c in suite.cases
        if (not cases or c.id in cases) and (split == "all" or c.split == split)
    ]
    if cases and set(cases) - {c.id for c in selected}:
        raise ValueError("任务不存在或不属于所选split")
    if not selected or not variants or not 1 <= repeats <= 10:
        raise ValueError("评测计划为空或重复次数无效")
    return [
        {
            "id": f"{variant}-{case.id}-{repeat}",
            "variant": variant,
            "case": case.model_dump(),
            "repeat": repeat,
        }
        for variant in variants
        for case in selected
        for repeat in range(repeats)
    ]


def prepare(root, bindings, settings):
    store = Store(root)
    sid = store.space("评测资料基线")["id"]
    assets = {}
    for alias, spec in bindings.items():
        source = Path(spec["path"])
        if not source.is_file():
            raise ValueError("缺少本机资料：" + str(source))
        if source.suffix.lower() == ".pdf":
            material = store.add_pdf(sid, source.name, source.read_bytes())
            mid = material["id"]
        elif source.suffix.lower() in {".txt", ".md"}:
            mid = store.add_material(sid, source.name, source.read_text())["id"]
        else:
            material = store.add_media_file(sid, source.name, source, source.suffix, move=False)
            mid = material["material_id"]
            # Explicitly prepared local preview, or let the existing media worker prepare it.
            if spec.get("preview"):
                shutil.copyfile(spec["preview"], root / material["preview_path"])
                with store.connect() as db:
                    db.execute("UPDATE media SET state='ready' WHERE material_id=?", (mid,))
            else:
                from learning_agent.jobs import Jobs
                from learning_agent.worker import Worker

                Jobs(store).enqueue("preview", sid, settings, mid)
                with Worker(settings.model_copy(update={"data_root": root})) as worker:
                    worker.drain()
                if store.media(sid, mid)["state"] != "ready":
                    raise ValueError("媒体准备失败：" + alias)
            if spec.get("captions"):
                from learning_agent.media import parse_captions

                path = Path(spec["captions"])
                segments = (
                    json.loads(path.read_text())["segments"]
                    if path.suffix == ".json"
                    else parse_captions(path.read_bytes(), material["duration"])
                )
                store.save_segments(
                    sid,
                    mid,
                    segments,
                    spec.get("transcript_source", "evaluation:uploaded_captions"),
                )
        store.set_source(
            sid,
            mid,
            {"url": spec.get("source_url"), "start_seconds": spec.get("source_start_seconds", 0)},
        )
        assets[alias] = {
            "id": mid,
            "name": source.name,
            "source_start_seconds": spec.get("source_start_seconds", 0),
            "transcript_source": spec.get("transcript_source"),
            "revision": store.material(sid, mid)["revision"],
            "source_bytes": source.stat().st_size,
            "source_modified_ns": source.stat().st_mtime_ns,
        }
    retriever = Retriever(store, settings)
    retriever.prepare(sid)
    return {
        "space_id": sid,
        "assets": assets,
        "index_version": INDEX_VERSION,
        "index_status": retriever.status(sid),
    }


def clone(source, target):
    target.mkdir(parents=True, exist_ok=False)
    with (
        sqlite3.connect(source / "workspace.db") as old,
        sqlite3.connect(target / "workspace.db") as new,
    ):
        old.backup(new)
    shutil.copytree(source / "materials", target / "materials")


async def checked(client, method, url, **kwargs):
    response = await client.request(method, url, **kwargs)
    if response.status_code >= 400:
        raise ValueError(f"评测操作失败 HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


async def actions(client, store, sid, cid, turn, saved, profile, ledger, key):
    for index, action in enumerate(turn["actions"]):
        # Actions are checkpointed independently, so --continue does not apply them twice.
        identity = f"{turn['id']}:{index}"
        if identity in saved.setdefault("actions", []):
            continue
        if saved.get("applying_action"):
            raise ValueError("初始化操作结果未确认，请使用新试验；不会自动重复修改状态")
        saved["applying_action"] = identity
        ledger.put(key, saved)
        values = action["values"]
        base = f"/api/conversations/{cid}"
        if action["kind"] == "learning":
            state = await checked(client, "GET", base + "/learning")
            await checked(
                client, "PATCH", base + "/learning", json={"revision": state["revision"], **values}
            )
        elif action["kind"] == "history":
            from learning_agent.context import ProtocolGroup

            # Synthetic initialization, not an evaluated model response.
            for row in values["turns"]:
                rid = store.start_run(cid, row["question"])
                groups = [
                    ProtocolGroup(
                        rid + ":input", "input", [{"role": "user", "content": row["question"]}]
                    ).dump(),
                    ProtocolGroup(
                        rid + ":answer", "answer", [{"role": "assistant", "content": row["answer"]}]
                    ).dump(),
                ]
                store.finish_run(
                    rid,
                    "completed",
                    row["answer"],
                    [],
                    archive={
                        "profile": profile,
                        "input": {"question": row["question"]},
                        "groups": groups,
                    },
                )
        else:
            if "topic_id" not in saved:
                topic = await checked(
                    client,
                    "POST",
                    f"/api/spaces/{sid}/learning-topics",
                    json={"title": "评测主题", "material_ids": []},
                )
                saved["topic_id"] = topic["id"]
            if action["kind"] == "memory_create":
                item = await checked(
                    client,
                    "POST",
                    f"/api/spaces/{sid}/memories",
                    json={"topic_id": saved["topic_id"], "kind": "preference", **values},
                )
                saved["memory"] = item
            elif action["kind"] == "memory_attach":
                state = await checked(client, "GET", base + "/context-settings")
                await checked(
                    client,
                    "PATCH",
                    base + "/context-settings",
                    json={
                        "revision": state["revision"],
                        "topic_id": saved["topic_id"],
                        "memory_enabled": values.get("enabled", True),
                    },
                )
            else:
                item = saved["memory"]
                url = f"/api/spaces/{sid}/memories/{item['id']}"
                if action["kind"] == "memory_forget":
                    await checked(client, "DELETE", url + f"?revision={item['revision']}")
                else:
                    saved["memory"] = await checked(
                        client,
                        "PATCH",
                        url,
                        json={
                            "revision": item["revision"],
                            **(
                                {"status": "disabled"}
                                if action["kind"] == "memory_disable"
                                else values
                            ),
                        },
                    )
        saved["actions"].append(identity)
        saved.pop("applying_action", None)
        ledger.put(key, saved)


async def run_experiment(
    output,
    suite,
    bindings,
    variants,
    *,
    base=None,
    cases=None,
    split="dev",
    repeats=1,
    max_turns=12,
    limits=None,
    continue_run=False,
    provider_factory=None,
):
    if max_turns < 1 or (limits and any(v < 0 for v in limits.values())):
        raise ValueError("回合上限必须为正，调用上限不能为负")
    output = Path(output).resolve()
    base = base or Settings()
    trials = plan(suite, variants, cases, split, repeats)
    limits = limits or {"main_model": 48, "summary": 4, "judge": 12}
    if output.exists() and not continue_run:
        raise ValueError("使用新的输出目录，或显式 --continue 接续原计划")
    ledger = Ledger(output / "experiment.db")
    definition = {
        "suite": suite.model_dump(),
        "bindings": bindings,
        "variants": variants,
        "trials": trials,
        "max_turns": max_turns,
        "limits": limits,
    }
    if ledger.get("definition") not in (None, definition):
        raise ValueError("接续计划与原计划不同，请使用新目录")
    if not ledger.get("definition"):
        ledger.put("definition", definition)
        ledger.put("limits", limits)
        try:
            commit = (
                await asyncio.to_thread(
                    subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
                )
            ).strip()
            dirty = bool(
                (
                    await asyncio.to_thread(
                        subprocess.check_output, ["git", "status", "--porcelain"], text=True
                    )
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = "unknown", None
        ledger.put(
            "manifest",
            {
                "id": output.name,
                "created_at": now(),
                "suite_id": suite.id,
                "suite_version": suite.version,
                "rubric_version": suite.rubric_version,
                "code_commit": commit,
                "dirty": dirty,
                "profile": profile_id(base),
                "limits": limits,
                "max_turns": max_turns,
            },
        )
    baseline = output / "corpus"
    if not ledger.get("corpus"):
        try:
            if baseline.exists():
                raise ValueError("基线准备未完整保存，请使用新输出目录；不复用半成品")
            ledger.put(
                "corpus",
                prepare(baseline, bindings, base.model_copy(update={"data_root": baseline})),
            )
        except (OSError, ValueError) as error:
            ledger.put("preparation_error", str(error))
            from learning_agent.evaluation.report import report

            return report(output)
    corpus = ledger.get("corpus")
    for trial in trials:
        key = "trial:" + trial["id"]
        saved = ledger.get(key, {**trial, "conversations": {}, "turns": {}, "actions": []})
        root = output / "trials" / trial["id"] / "workspace"
        if not root.exists():
            clone(baseline, root)
        config = configuration(
            base,
            {**variants[trial["variant"]], **trial["case"]["settings"]},
            data_root=root,
            evaluation_budget_path=output / "experiment.db",
        )
        profile = {
            "model": profile_id(config),
            "route_policy": config.route_policy,
            "retrieval_mode": config.retrieval_mode,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "context_budget_tokens": config.context_budget_tokens,
            "max_steps": config.max_steps,
            "stream_answer": config.stream_answer,
        }
        if saved.get("profile") not in (None, profile):
            raise ValueError("接续试验的有效配置已变化")
        saved["profile"] = profile
        app = create_app(config, provider_factory(config) if provider_factory else None)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://evaluation"
            ) as client,
        ):
            blocked = False
            for turn in trial["case"]["turns"]:
                item = saved["turns"].get(turn["id"], {})
                if item.get("trace", {}).get("complete"):
                    blocked = item["trace"]["state"] not in {"completed", "budget_exhausted"}
                    continue
                if blocked:
                    item["status"] = "blocked"
                    saved["turns"][turn["id"]] = item
                    continue
                if not item.get("request_id"):
                    count = sum(
                        bool(t.get("request_id"))
                        for r in ledger.records("trial:")
                        for t in r["turns"].values()
                    )
                    if count >= max_turns:
                        saved["turns"][turn["id"]] = {"status": "stopped_budget"}
                        blocked = True
                        continue
                cid = saved["conversations"].get(turn["conversation"])
                if not cid:
                    cid = (
                        await checked(
                            client, "POST", f"/api/spaces/{corpus['space_id']}/conversations"
                        )
                    )["id"]
                    saved["conversations"][turn["conversation"]] = cid
                    ledger.put(key, saved)
                try:
                    await actions(
                        client,
                        app.state.store,
                        corpus["space_id"],
                        cid,
                        turn,
                        saved,
                        profile_id(config),
                        ledger,
                        key,
                    )
                except ValueError as error:
                    saved["turns"][turn["id"]] = {
                        "status": "preparation_failed",
                        "error": str(error),
                    }
                    ledger.put(key, saved)
                    blocked = True
                    continue
                ledger.put(key, saved)
                if not item.get("request_id"):
                    item = {"request_id": str(uuid4()), "status": "running", "started_at": now()}
                    saved["turns"][turn["id"]] = item
                    ledger.put(key, saved)
                payload = {
                    "content": turn["input"]["question"],
                    "client_request_id": item["request_id"],
                }
                if turn["input"]["focus"]:
                    focus = dict(turn["input"]["focus"])
                    focus["material_id"] = corpus["assets"][focus.pop("asset")]["id"]
                    payload["focus"] = focus
                if turn["input"]["feedback"]:
                    feedback = dict(turn["input"]["feedback"])
                    target = feedback.pop("target_turn")
                    feedback["target_run_id"] = saved["turns"][target]["run_id"]
                    payload["feedback"] = feedback
                start = time.monotonic()
                rid = item.get("run_id")
                if not rid:
                    result = await checked(
                        client, "POST", f"/api/conversations/{cid}/runs", json=payload
                    )
                    rid = result["run_id"]
                    item.update(run_id=rid, accepted_s=time.monotonic() - start)
                    ledger.put(key, saved)
                # Runtime startup may safely recover a saved response. Unknown results remain paused.
                deadline = time.monotonic() + config.timeout_seconds * 3 + 20
                while True:
                    run = await checked(client, "GET", f"/api/runs/{rid}")
                    if run["state"] not in {"queued", "running", "cancelling"}:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("等待评测运行超时；不自动重发")
                    await asyncio.sleep(0.05)
                trace = collect(app.state.store, rid)
                item.update(
                    trace=trace,
                    status=trace["state"]
                    if trace["state"] in {"completed", "budget_exhausted"}
                    else "interrupted"
                    if trace["state"] == "interrupted"
                    else "stopped_budget"
                    if any(e["type"] == "evaluation.budget_exhausted" for e in trace["events"])
                    else "run_failed",
                )
                ledger.put(key, saved)
                save_json(root.parent / (turn["id"] + "-trace.json"), trace)
                blocked = item["status"] not in {"completed", "budget_exhausted"}
                print(
                    json.dumps(
                        {"trial": trial["id"], "turn": turn["id"], "status": item["status"]}
                    ),
                    flush=True,
                )
        ledger.put(key, saved)
    from learning_agent.evaluation.report import report

    return report(output)
