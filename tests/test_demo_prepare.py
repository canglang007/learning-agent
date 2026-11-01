import runpy
import sqlite3
import sys
from pathlib import Path

from learning_agent.store import Store


def test_prepare_demo_creates_self_contained_workspace(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    script = Path(__file__).parents[1] / "examples" / "prepare_demo.py"
    monkeypatch.setattr(sys, "argv", ["prepare_demo.py", "--output", str(root)])
    runpy.run_path(str(script), run_name="__main__")

    store = Store(root)
    spaces = store.spaces()
    assert len(spaces) == 1
    sid = spaces[0]["id"]
    materials = store.materials(sid)
    assert {item["name"] for item in materials} == {
        "matrix-notes.md",
        "matrix-worksheet.pdf",
    }
    conversations = store.conversations(sid)
    assert len(conversations) == 1
    messages = store.messages(conversations[0]["id"])
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "1×0 + 2×1 + 3×0 = 2" in messages[-1]["content"]
    with sqlite3.connect(root / "workspace.db") as db:
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
