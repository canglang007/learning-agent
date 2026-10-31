import asyncio
import io

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session
from reportlab.pdfgen import canvas

from learning_agent.config import Settings
from learning_agent.mcp_server import McpWorkspace, create_server
from learning_agent.retrieval import Retriever
from learning_agent.store import Store


def make_workspace(tmp_path):
    store = Store(tmp_path)
    space = store.space("学习")
    store.add_material(space["id"], "notes.md", "CS336 的归一化在分支内完成，跳连不经过归一化。")
    settings = Settings(_env_file=None, data_root=tmp_path, retrieval_mode="bm25")
    Retriever(store, settings).index.chunks(space["id"])
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "PDF page text about RMSNorm and residual paths")
    pdf.save()
    material = store.add_pdf(space["id"], "slides.pdf", buffer.getvalue())
    return space, material, McpWorkspace(tmp_path, space["id"])


def test_workspace_lists_searches_and_reads_page(tmp_path):
    space, material, workspace = make_workspace(tmp_path)

    listed = workspace.list_materials()
    assert listed["space_id"] == space["id"]
    assert {item["name"] for item in listed["materials"]} == {"notes.md", "slides.pdf"}

    found = workspace.search_materials("CS336 归一化")
    assert found["retrieval"] == "bm25_read_only"
    assert found["hits"]
    assert found["hits"][0]["name"] == "notes.md"

    page = workspace.read_page(material["id"], 1)
    assert page["name"] == "slides.pdf" and page["page"] == 1
    assert page["page_url"].endswith("/pages/1/image")

    with pytest.raises(LookupError):
        workspace.read_page(material["id"], 999)


def test_workspace_requires_space_when_multiple_spaces(tmp_path):
    space, _material, _workspace = make_workspace(tmp_path)
    store = Store(tmp_path)
    store.space("另一个空间")
    workspace = McpWorkspace(tmp_path)
    with pytest.raises(ValueError, match="space_id"):
        workspace.list_materials()
    assert workspace.list_materials(space["id"])["space_id"] == space["id"]


def test_mcp_server_lists_tools_and_calls_search(tmp_path):
    space, material, _workspace = make_workspace(tmp_path)
    server = create_server(tmp_path, space["id"])

    async def exercise():
        async with create_connected_server_and_client_session(server) as session:
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "list_materials",
                "search_materials",
                "read_page",
            }
            found = await session.call_tool("search_materials", {"query": "CS336 归一化"})
            assert not found.isError
            page = await session.call_tool(
                "read_page", {"material_id": material["id"], "page": 1}
            )
            assert not page.isError

    asyncio.run(exercise())
