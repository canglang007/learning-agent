# MCP 只读服务器

更新于：2026-11-01 00:33:05（Asia/Shanghai，UTC+08:00）。

Learning Agent 提供一个可选的 MCP（Model Context Protocol）stdio 服务器，把本地资料目录暴露给兼容的 MCP 客户端。它是**只读适配层**，不是新的执行引擎：不调用模型、不上传、不修改资料、不改学习状态。

## 安装与启动

```bash
uv sync --extra mcp
uv run learning-agent-mcp --data-root ./data --space <space_id>
```

`--space` 可以填资料空间 ID 或名称。只有一个空间时可以省略；多个空间时工具需要显式传 `space_id`。

查看可用空间：

```bash
uv run learning-agent-mcp --data-root ./data --list-spaces
```

也可以设置 `LA_MCP_SPACE`，或把它写进 `.env`：

```dotenv
LA_MCP_SPACE=
```

## 客户端配置

不同 MCP 客户端的配置字段不同；核心是让客户端用 stdio 启动：

```json
{
  "mcpServers": {
    "learning-agent": {
      "command": "/absolute/path/to/learning-agent/.venv/bin/learning-agent-mcp",
      "args": ["--data-root", "/absolute/path/to/learning-agent/data", "--space", "<space_id>"]
    }
  }
}
```

保存配置后重启客户端；服务器不会自动启动 Learning Agent 的 API 或 worker。

## 工具

| 工具 | 参数 | 返回 |
| --- | --- | --- |
| `list_materials` | `space_id` 可选 | 资料 ID、名称、格式、PDF 页数、媒体/字幕状态 |
| `search_materials` | `query`，`space_id`/`material_id` 可选 | 本地 BM25 前 5 个分段，含页码/行号/时间、摘要与回访链接 |
| `read_page` | `material_id`、`page`，`space_id` 可选 | PDF 指定物理页的提取文字、行数、是否有文字、页面链接 |

### 只读边界

- 不提供上传、删除、修改、学习状态或模型调用工具。
- `read_page` 不渲染图片、不返回原始文件路径；它带 `text_source`（`text` / `ocr` / `none` / `blank`）。`ocr` 是已识别过的扫描页文字，`none` 是还没识别过的页，`blank` 是识别过仍无文字。图表与公式排版仍需在网页查看。
- 只读服务不写库，因此不在 MCP 里做按需识别：`none` 的页需要在网页里读一次，识别结果会写回该页文字；要让整本可检索则是网页端的整本识别动作。
- `search_materials` 只读取已存在的 `chunks`，不建立索引、不调用向量服务；没有分段时明确返回空结果和原因。
- 所有资料 ID 都经过空间归属校验，不接受任意文件路径。
- MCP 客户端看到的是本地资料适配层；网页仍是完整交互入口。

## 本机演示

使用本机 `data/acceptance/optimization.pdf`（119 页）建立独立演示空间，分段由网页/评测准备，MCP 本身不写索引：

```text
list_materials
  -> optimization.pdf, PDF, 119 pages

search_materials("learning rate")
  -> 5 hits, bm25_read_only
  -> 第一名：optimization.pdf 物理第 77 页，
     "Learning rate schedules ..."

read_page(material_id, 77)
  -> has_text: true, total_lines: 4
```

中文查询英文 PDF 可能不命中，因为 MCP 只做 BM25 词法检索；请用文档语言查询，或先在网页准备多语言向量。MCP 不代替网页的视觉/向量能力。


## 为什么要单独做 MCP

- 让外部 Agent 能复用同一份本地资料，而不是复制一份 RAG 数据。
- 用协议层暴露工具 Schema、参数和错误，而不是把数据库直接交给外部程序。
- 保持只读，避免在外部 Agent 中产生上传、删除或状态修改的副作用。

## 测试

```bash
uv sync --extra mcp --extra dev
uv run pytest -q tests/test_mcp.py
```

测试使用内存 MCP 会话验证工具发现、搜索和 PDF 页读取，不启动外部客户端。
