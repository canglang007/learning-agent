"""Read-only MCP stdio server for inspecting a local learning-agent workspace."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from learning_agent.retrieval import bm25, course_scope
from learning_agent.store import Store

SERVER_INSTRUCTIONS = """Learning Agent 的只读资料服务器。
只能查看资料目录、搜索已建立的分段、读取 PDF 页面文字；不能上传、修改、删除或调用模型。
涉及页面图像、图表、视频画面或字幕时，请让用户在 Learning Agent 网页中查看。"""


class McpWorkspace:
    def __init__(self, data_root: str | Path, space: str | None = None) -> None:
        self.root = Path(data_root).expanduser().resolve()
        if not (self.root / "workspace.db").is_file():
            raise ValueError(f"找不到本地工作区：{self.root / 'workspace.db'}")
        self.store = Store(self.root)
        self.spaces = self.store.spaces()
        self.default_space = self._resolve(space) if space else (
            self.spaces[0]["id"] if len(self.spaces) == 1 else None
        )

    def _resolve(self, value: str) -> str:
        for item in self.spaces:
            if value in {item["id"], item["name"]}:
                return item["id"]
        raise ValueError("资料空间不存在：" + value)

    def space_id(self, value: str | None = None) -> str:
        if value:
            return self._resolve(value)
        if self.default_space:
            return self.default_space
        names = ", ".join(f"{item['name']}={item['id']}" for item in self.spaces) or "无"
        raise ValueError("有多个资料空间，请传入 space_id；可选：" + names)

    def spaces_view(self) -> dict[str, Any]:
        return {
            "spaces": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "default": item["id"] == self.default_space,
                }
                for item in self.spaces
            ]
        }

    def list_materials(self, space_id: str | None = None) -> dict[str, Any]:
        sid = self.space_id(space_id)
        items = self.store.materials(sid)
        return {
            "space_id": sid,
            "materials": [
                {
                    key: item.get(key)
                    for key in (
                        "id",
                        "name",
                        "format",
                        "revision",
                        "page_count",
                        "media_state",
                        "transcript_state",
                    )
                }
                for item in items
            ],
            "detail": "只返回资料索引信息；正文请用 search_materials/read_page。",
        }

    def search_materials(
        self,
        query: str,
        space_id: str | None = None,
        material_id: str | None = None,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        if len(query) > 300:
            raise ValueError("query 最长 300 字符")
        sid = self.space_id(space_id)
        if material_id:
            self.store.material(sid, material_id)
        chunks = self._chunks(sid, material_id)
        if not chunks:
            return {
                "space_id": sid,
                "query": query,
                "hits": [],
                "retrieval": "bm25_read_only",
                "detail": "该资料空间还没有已建立的分段；请先在网页打开资料或准备索引。MCP 不写入索引。",
            }
        scoped, search_query, scoped_ids = course_scope(query, chunks)
        scores = bm25(search_query, [chunk["excerpt"] for chunk in scoped])
        ranking = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )
        hits, per_page = [], Counter()
        for index in ranking:
            chunk = scoped[index]
            key = (chunk.get("id"), chunk.get("page") or chunk.get("ordinal"))
            if per_page[key] >= 2:
                continue
            per_page[key] += 1
            hits.append(
                {
                    **{
                        name: chunk[name]
                        for name in (
                            "chunk_id",
                            "material_id",
                            "id",
                            "name",
                            "page",
                            "start_line",
                            "end_line",
                            "start_seconds",
                            "end_seconds",
                        )
                        if name in chunk
                    },
                    "excerpt": str(chunk.get("excerpt", ""))[:900],
                    "score": scores[index],
                    "page_url": chunk.get("page_url"),
                    "source_url": chunk.get("source_url"),
                }
            )
            if len(hits) == 5:
                break
        return {
            "space_id": sid,
            "query": query,
            "hits": hits,
            "retrieval": "bm25_read_only",
            "scope": {"material_ids": scoped_ids},
            "detail": "MCP 只做本地 BM25，不调用向量服务；排名不是正确性概率。",
        }

    def read_page(
        self, material_id: str, page: int, space_id: str | None = None
    ) -> dict[str, Any]:
        if page < 1:
            raise ValueError("page 从 1 开始")
        sid = self.space_id(space_id)
        result = self.store.page(sid, material_id, page)
        text = result.get("text", "")
        source = result.get("text_source") or "text"
        detail = "只返回 PDF 页内文字；图表、公式排版和扫描页图像需要在网页查看。"
        if source == "ocr":
            detail = (
                "该页文字来自 OCR（扫描页），可能有错字、公式符号错误或顺序错误；"
                "图表与公式排版需要在网页看图像。"
            )
        elif source == "none":
            detail = "该页没有可提取文字（扫描页或纯图页），文字通道拿不到内容，需要在网页查看图像。"
        return {
            "space_id": sid,
            "material_id": material_id,
            "name": result.get("name"),
            "page": page,
            "text": text[:12000],
            "total_lines": len(text.splitlines()),
            "truncated": len(text) > 12000,
            "has_text": bool(text.strip()),
            "text_source": source,
            "page_url": f"/api/spaces/{sid}/materials/{material_id}/pages/{page}/image",
            "detail": detail,
        }

    def _chunks(self, space_id: str, material_id: str | None = None) -> list[dict[str, Any]]:
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT c.payload FROM chunks c JOIN materials m ON m.id=c.material_id "
                "WHERE m.space_id=?"
                + (" AND m.id=?" if material_id else "")
                + " ORDER BY m.rowid,c.ordinal",
                (space_id, material_id) if material_id else (space_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


def create_server(data_root: str | Path, space: str | None = None) -> FastMCP:
    workspace = McpWorkspace(data_root, space)
    server = FastMCP("learning-agent", instructions=SERVER_INSTRUCTIONS)

    @server.tool()
    def list_materials(space_id: str | None = None) -> dict[str, Any]:
        """列出资料空间中的资料；space_id 可省略，单空间工作区会使用默认空间。"""
        try:
            return workspace.list_materials(space_id)
        except (LookupError, ValueError) as error:
            raise ValueError(str(error)) from None

    @server.tool()
    def search_materials(
        query: str,
        space_id: str | None = None,
        material_id: str | None = None,
    ) -> dict[str, Any]:
        """在已有分段上做只读 BM25 搜索，返回前 5 个候选和页码/时间位置。"""
        try:
            return workspace.search_materials(query, space_id, material_id)
        except (LookupError, ValueError) as error:
            raise ValueError(str(error)) from None

    @server.tool()
    def read_page(
        material_id: str, page: int, space_id: str | None = None
    ) -> dict[str, Any]:
        """读取 PDF 指定物理页的提取文字；不渲染图片，不返回原始文件路径。"""
        try:
            return workspace.read_page(material_id, page, space_id)
        except (LookupError, ValueError) as error:
            raise ValueError(str(error)) from None

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning Agent 的只读 MCP stdio 服务器")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LA_DATA_ROOT", "data")),
    )
    parser.add_argument("--space", default=os.environ.get("LA_MCP_SPACE"))
    parser.add_argument("--list-spaces", action="store_true")
    args = parser.parse_args()
    try:
        server = create_server(args.data_root, args.space)
        if args.list_spaces:
            workspace = McpWorkspace(args.data_root, args.space)
            print(json.dumps(workspace.spaces_view(), ensure_ascii=False, indent=2))
            return
    except (OSError, ValueError, LookupError) as error:
        parser.error(str(error))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
