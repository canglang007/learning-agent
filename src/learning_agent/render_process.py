"""Short-lived render process, owned process group, parent-death pipe, staged outputs."""

import asyncio
import json
import os
import shutil
import signal
import sys
import threading
from pathlib import Path
from uuid import uuid4

from learning_agent.provider import ModelError, asset_info


async def render_tool(control, tools, name, arguments, image_budget=3):
    attempt = tools.store.root.resolve() / ".runtime-attempts" / str(uuid4())
    attempt.mkdir(parents=True)
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "learning_agent.render_process",
            str(attempt),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        request = {
            "root": str(tools.store.root.resolve()),
            "space_id": tools.space_id,
            "vision": tools.vision,
            "name": name,
            "arguments": arguments,
            "image_budget": image_budget,
        }
        process.stdin.write((json.dumps(request) + "\n").encode())
        await process.stdin.drain()
        await process.wait()
        if process.returncode:
            raise ModelError("资料渲染进程未完成，请重新查看资料。")
        result = json.loads((attempt / "result.json").read_text())
        control.check()
        control.validate()
        control.verify_observed()
        images = ([result["image"]] if "image" in result else []) + result.get("images", [])
        # Publication is short, owner-checked, and never exposes the child's partial file.
        with tools.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            tools.store.execution_owner(db, control.run_id, control.owner)
            control.verify_observed()
            for image in images:
                candidate = attempt / image["asset_path"]
                if candidate.exists():
                    asset_info(attempt, image)
                    target = tools.store.root / image["asset_path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    candidate.replace(target)
                asset_info(tools.store.root, image)
        return result
    finally:
        if process:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()
            if process.stdin:
                process.stdin.close()
        shutil.rmtree(attempt, ignore_errors=True)


def child():
    from learning_agent.assets import staging_directory
    from learning_agent.media import _render_process_group
    from learning_agent.store import Store
    from learning_agent.tools import MaterialTools

    attempt = Path(sys.argv[1])
    request = json.loads(sys.stdin.readline())

    # Parent SIGKILL closes its pipe. All this child's FFmpeg descendants share its group.
    def watch_parent():
        while os.read(sys.stdin.fileno(), 1):
            pass
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_parent, daemon=True).start()
    staging_directory.set(attempt)
    _render_process_group.set(True)
    tools = MaterialTools(Store(Path(request["root"])), request["space_id"], request["vision"])
    if request["name"] == "_restore_frame":
        result = {"image": tools.store.media_frame(request["space_id"], **request["arguments"])}
    elif request["name"] == "_restore_page":
        result = {"image": tools.store.page_image(request["space_id"], **request["arguments"])}
    else:
        result = tools.execute(
            request["name"], request["arguments"], image_budget=request["image_budget"]
        )
    (attempt / "result.json").write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    child()
