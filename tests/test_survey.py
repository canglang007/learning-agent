"""试用问卷：答案落在同一个库，能汇总也能导出。"""

import json

from client_helpers import TestClient, close_for

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.store import Store
from learning_agent.survey import export


def settings(root):
    item = Settings(
        _env_file=None,
        data_root=root,
        retrieval_mode="bm25",
        embedding_provider="disabled",
    )
    Store(root)
    return item


def test_page_renders(tmp_path):
    root = tmp_path / "survey"
    client = TestClient(create_app(settings(root)))
    try:
        page = client.get("/survey")
        assert page.status_code == 200
        assert "试用反馈" in page.text
    finally:
        close_for(root)


def test_submit_then_summarise(tmp_path):
    root = tmp_path / "collect"
    client = TestClient(create_app(settings(root)))
    try:
        assert client.post("/api/survey", json={"rating": 4, "role": "研究生"}).status_code == 201
        assert client.post("/api/survey", json={"rating": 2, "fabricated": "有"}).status_code == 201
        stats = client.get("/api/survey").json()
        assert stats["responses"] == 2
        assert stats["rating_mean"] == 3.0
        assert stats["fabricated_reports"] == 1
    finally:
        close_for(root)


def test_rating_out_of_range_is_rejected(tmp_path):
    root = tmp_path / "bounds"
    client = TestClient(create_app(settings(root)))
    try:
        assert client.post("/api/survey", json={"rating": 9}).status_code == 422
        assert client.post("/api/survey", json={"rating": "嗯"}).status_code == 422
        assert client.get("/api/survey").json()["responses"] == 0
    finally:
        close_for(root)


def test_export_keeps_original_answers(tmp_path):
    root = tmp_path / "export"
    client = TestClient(create_app(settings(root)))
    try:
        client.post("/api/survey", json={"rating": 5, "stuck": "公式那步跳太快"})
    finally:
        close_for(root)
    output = tmp_path / "survey.json"
    payload = export(Store(root), output)
    assert payload["summary"]["responses"] == 1
    saved = json.loads(output.read_text())
    assert saved["responses"][0]["stuck"] == "公式那步跳太快"
