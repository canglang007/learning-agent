"""Owned child process for RT3 kill-at-boundary tests; never contacts a network."""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from test_feedback import deliver
from test_streaming import Bytes, chat_events, event

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.provider import OpenAIProvider


async def main():
    root = Path(os.environ["RT_ROOT"])
    target = os.environ["RT_TARGET"]
    settings = Settings(_env_file=None, **json.loads((root / "settings.json").read_text()))

    def handler(request):
        with (root / "calls").open("a") as file:
            file.write("request\n")
        chunks = [event(e) for e in chat_events(deliver("可恢复的完整讲解😀"))] + [event("[DONE]")]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Bytes(chunks)
        )

    app = create_app(settings, OpenAIProvider(settings, httpx.MockTransport(handler)))
    store = app.state.store
    method = "write_draft" if target == "partial" else "complete_operation"
    original = getattr(store, method)

    def boundary(*args, **kwargs):
        result = original(*args, **kwargs)
        if not (root / "boundary").exists():
            (root / "boundary").write_text(target)
            while True:
                time.sleep(0.05)
        return result

    setattr(store, method, boundary)
    async with app.router.lifespan_context(app):
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
