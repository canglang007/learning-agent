import asyncio
import base64
import io
import json
import sqlite3

import httpx
import pytest
from client_helpers import TestClient, close_for
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from test_learning import Scripted, call

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.provider import OpenAIProvider
from learning_agent.store import Store
from learning_agent.tools import MaterialTools, definitions


def pdf_bytes():
    buffer = io.BytesIO()
    doc = canvas.Canvas(buffer, pagesize=(500, 400))
    doc.drawString(30, 350, "Page one: chain rule; Y = XW")
    doc.showPage()
    # All chart content is a raster: text extraction cannot reveal the answer.
    image = Image.new("RGB", (400, 250), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(30, 20), (30, 220), (380, 220)], fill="black", width=3)
    draw.line([(30, 50), (120, 110), (250, 180), (380, 200)], fill="red", width=4)
    draw.text((100, 20), "Loss against step", fill="black")
    doc.drawImage(ImageReader(image), 40, 80, width=400, height=250)
    doc.save()
    return buffer.getvalue()


def prepared(tmp_path, model=None, **settings):
    config = Settings(_env_file=None, data_root=tmp_path, **settings)
    model = model or Scripted([])
    close_for(tmp_path)
    client = TestClient(create_app(config, model))
    space = client.post("/api/spaces", json={"content": "pdf study"}).json()["id"]
    uploaded = client.post(
        f"/api/spaces/{space}/materials",
        files={"file": ("course.pdf", pdf_bytes(), "application/pdf")},
    )
    assert uploaded.status_code == 201
    material = uploaded.json()["id"]
    conv = client.post(f"/api/spaces/{space}/conversations").json()["id"]
    return client, space, material, conv, model


def test_pdf_preserves_pages_search_and_raster_only_page(tmp_path):
    client, space, material, _, _ = prepared(tmp_path)
    base = f"/api/spaces/{space}/materials/{material}"
    pages = client.get(base + "/pages").json()
    assert len(pages) == 2
    assert pages[0]["text_characters"] > 0 and pages[1]["text_characters"] == 0
    assert "chain rule" in client.get(base + "/pages/1").json()["text"]
    image = client.get(base + "/pages/2/image")
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    tools = MaterialTools(Store(tmp_path), space)
    hit = tools.execute("search_materials", {"query": "chain rule"})["hits"][0]
    assert hit["page"] == 1
    nav = tools.execute("material_outline", {"material_id": material})
    assert len(nav["entries"]) == 2 and nav["entries"][1]["chunk_id"] is None
    assert tools.execute("read_page", {"material_id": material, "page": 2})["has_text"] is False


def test_visual_tool_attaches_image_after_all_tool_outputs_and_persists_refs(tmp_path):
    client, _space, material, conv, model = prepared(tmp_path)
    first = call("read_page", {"material_id": material, "page": 2})
    first["tool_calls"].extend(
        call("inspect_page", {"material_id": material, "page": 2, "region": None}, "image-call")[
            "tool_calls"
        ]
    )
    model.responses = [
        first,
        {"role": "assistant", "content": "曲线逐渐下降。"},
        {"role": "assistant", "content": "换个方式解释。"},
    ]
    url = f"/api/conversations/{conv}/messages"
    result = client.post(
        url, json={"content": "解释这张图", "focus": {"material_id": material, "page": 2}}
    ).json()
    assert result["state"] == "completed"
    next_input = model.inputs[1]
    assert [m["role"] for m in next_input[-3:]] == ["tool", "tool", "user"]
    attachment = next_input[-1]["attachments"][0]
    assert (tmp_path / attachment["asset_path"]).is_file()
    assert "base64" not in json.dumps(client.get(f"/api/runs/{result['run_id']}").json())
    client.post(url, json={"content": "我没懂"})
    assert any(m.get("attachments") for m in model.inputs[-1])
    with sqlite3.connect(tmp_path / "workspace.db") as db:
        protocol = db.execute("SELECT protocol FROM conversations WHERE id=?", (conv,)).fetchone()[
            0
        ]
    assert "base64" not in protocol


def test_pdf_regions_errors_and_workspace_scope(tmp_path):
    client, space, material, _conv, _ = prepared(tmp_path)
    tools = MaterialTools(Store(tmp_path), space)
    args = {"material_id": material, "page": 2, "region": None}
    full = tools.execute("inspect_page", args)
    crop = tools.execute("inspect_page", {**args, "region": [0.1, 0.1, 0.7, 0.7]})
    assert crop["image"]["width"] < full["image"]["width"]
    assert (
        tools.execute("inspect_page", {**args, "region": [0.9, 0.1, 0.2, 0.7]})["error"]
        == "invalid_region"
    )
    missing = tools.execute("inspect_page", {**args, "page": 3})
    assert missing["error"] == "material_not_found"
    # 错误文案要给出可执行的下一步，否则模型只会换一个猜的ID重试。
    assert "list_materials" in missing["detail"]
    other = client.post("/api/spaces", json={"content": "other"}).json()["id"]
    assert client.get(f"/api/spaces/{other}/materials/{material}/pages/1/image").status_code == 404
    other_conv = client.post(f"/api/spaces/{other}/conversations").json()["id"]
    assert (
        client.post(
            f"/api/conversations/{other_conv}/messages",
            json={"content": "see", "focus": {"material_id": material, "page": 1}},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/spaces/{space}/materials", files={"file": ("broken.pdf", b"bad")}
        ).status_code
        == 422
    )
    disabled = MaterialTools(Store(tmp_path), space, vision=False)
    assert disabled.execute("inspect_page", args)["error"] == "vision_disabled"
    assert "inspect_page" not in {tool["name"] for tool in definitions(False)}


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_provider_sends_real_png_in_both_protocols_without_internal_paths(tmp_path, protocol):
    Image.new("RGB", (30, 30), "red").save(tmp_path / "page.png")
    seen = []

    def transport(request):
        seen.append(json.loads(request.content))
        if protocol == "chat_completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "red"}}]})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "red"}]}
                ],
            },
        )

    async def run():
        provider = OpenAIProvider(
            Settings(_env_file=None, data_root=tmp_path, protocol=protocol),
            httpx.MockTransport(transport),
        )
        await provider.step(
            [
                {
                    "role": "user",
                    "content": "图中颜色？",
                    "attachments": [{"asset_path": "page.png"}],
                }
            ],
            "",
            [],
            False,
        )

    asyncio.run(run())
    payload = seen[0]
    content = (
        payload["messages"][-1]["content"]
        if protocol == "chat_completions"
        else payload["input"][-1]["content"]
    )
    url = (
        content[-1]["image_url"]["url"]
        if protocol == "chat_completions"
        else content[-1]["image_url"]
    )
    assert base64.b64decode(url.split(",")[1]).startswith(b"\x89PNG")
    assert "asset_path" not in json.dumps(payload)


def test_image_budget_remains_bounded(tmp_path):
    client, _, material, conv, model = prepared(tmp_path, max_images_per_turn=1)
    model.responses = [
        call("inspect_page", {"material_id": material, "page": 1, "region": None}),
        call("inspect_page", {"material_id": material, "page": 2, "region": None}, "other"),
        {"role": "assistant", "content": "本轮不能继续看图，还需要确认。"},
    ]
    result = client.post(f"/api/conversations/{conv}/messages", json={"content": "读两个图"}).json()
    assert result["state"] == "completed"
    assert json.loads(model.inputs[-1][-1]["content"])["error"] == "image_budget_exhausted"


def test_existing_text_database_migrates_without_losing_material(tmp_path):
    path = tmp_path / "workspace.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE materials(id TEXT PRIMARY KEY,space_id TEXT,name TEXT,text TEXT)")
        db.execute("INSERT INTO materials VALUES ('old','space','notes.md','original text')")
    store = Store(tmp_path)
    assert store.material("space", "old")["text"] == "original text"
    assert store.material("space", "old")["format"] == "text"
