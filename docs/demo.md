# 3–5 分钟真实课程多模态演示

更新于：2026-11-14 20:55（Asia/Shanghai，UTC+08:00）。

这套演示使用**真实课程 PDF 和真实课程视频片段**，不是合成幻灯片。它实际走通：PDF 页面图像 → 视频字幕 → 视频抽样帧 → 带来源回答 → 网页回跳。课程素材只保留在本机，不随仓库分发。

## 准备

需要本机已有课程素材（`data/acceptance/real-courses/`）：

- `cs336-architecture.pdf`：真实课程讲义，演示使用第 10 页。
- `cs336-prenorm.mp4`：真实课程视频片段，198 秒。
- `cs336-prenorm-asr.json`：该片段的本地 ASR 结果，作为带时间字幕。
- `ntu-tool-use.mp4`、`ntu-tool-use-asr.json`：另一门真实课程视频及字幕，用来展示资料空间不只一份视频。
- `optimization.pdf`：默认路径 `data/acceptance/optimization.pdf` 的第二份真实 PDF。
- `ffmpeg` / `ffprobe`：生成回放和抽样帧。
- 已配置讲解模型；准备阶段不调用模型，提问阶段会计费。

`prepare_course_demo.py` 还会生成合成 `worksheet.pdf` 和 `study-notes.md` 作为辅助资料；它们不是课程证据。

素材来源：CS336 课程视频（<https://www.youtube.com/watch?v=ptFiH_bHnJw>），`prepare_course_demo.py` 记录的原视频偏移为 353 秒；演示中的时间都是片段相对时间。课程视频和讲义不随仓库分发。

新建独立演示空间：

```bash
uv run python examples/prepare_course_demo.py \
  --output data/demo \
  --workspace data/demo/workspace-multimodal
./start.sh --data-root data/demo/workspace-multimodal --evaluation-root data/evaluations
```

如果已经有 `data/demo/workspace`，可以直接启动：

```bash
./start.sh --data-root data/demo/workspace --evaluation-root data/evaluations
```

打开 <http://127.0.0.1:8320>。素材不存在时，使用[离线合成演示](demo-offline.md)，或在网页上传自己的 PDF/视频；不要用合成幻灯片冒充真实课程视频。

## 演示顺序

### 0:00–0:40 真实资料与目标

- 打开资料空间，展示 `cs336-architecture.pdf`（68 页）、`cs336-prenorm.mp4`（198 秒，字幕 ready）和另一门课程视频。
- 说明这是真实课程片段，只在本机保存；不是预写动画，也不随仓库公开。
- 目标不是“检索一段文本”，而是让 Agent 实际看页面、读字幕、抽帧，再讲清并给出可回查位置。

### 0:40–2:20 真实多模态提问

> 请实际查看 cs336-architecture.pdf 第10页，并结合 cs336-prenorm.mp4 的20到75秒，讲清 pre-norm 和 post-norm 的残差直通区别。三四句话即可，给一个可回听窗口。

观察并讲解：

- 模型先尝试调用页面/媒体工具；如果它还不知道资料 ID，会收到 `material_not_found`，随后调用 `list_materials` 恢复。这不是脚本预设，而是真实工具循环。
- 恢复后实际调用 `inspect_page` 读取第 10 页，调用 `inspect_media` 检查 20–75 秒的**字幕 + 抽样画面**。
- 本轮第 4 步模型输入包含 3 张实际图片：PDF 第 10 页 + 视频 33.750 秒帧 + 61.250 秒帧；不是把整段视频塞进模型。
- 回答给出 post-norm `LN(x_l + F(x_l))` 与 pre-norm `x_l + F(LN(x_l))` 的区别，并引用 `[cs336-architecture.pdf:p10]`、`[cs336-prenorm.mp4:00:25-00:56]`。

### 2:20–3:20 来源回跳

- 点击“查看 cs336-architecture.pdf 第10页”，阅读器打开真实页面图像。
- 点击“回听 cs336-prenorm.mp4 0:20–1:15”，播放器定位到工具实际检查的 20–75 秒；回答引用的是其中 00:25–00:56 的子窗口。
- 指出回答引用的是模型实际检查过的 20–75 秒内的子窗口，不是用户问题里的时间字符串。
- 展开“查看本轮工具与执行状态”，区分：字幕来源 `local_asr:base:reused_acceptance`、抽样帧时间、页面图像；抽样帧不等于连续观看，字幕也不等于直接听音频。

### 3:20–4:20 运行记录与评测

- 展开“查看本轮工具与执行状态”，展示路由 `agent`、工具步骤、图片数量和来源引用。
- 打开 `/evaluation`，说明执行状态、Token/估算输入、内容复核是不同维度；历史报告是开发复核，不是真人学习收益。
- 本轮真实记录的本地证据：`data/demo/multimodal-run-2026-11-14.json`（不提交）。

### 4:20–5:00 边界

- 课程视频和讲义属于原课程；仓库不重新分发，公开演示前要替换为有权使用的素材。
- ASR 可能有识别错误；当前是字幕 + 抽样帧，不是逐帧观看或直接听音频。
- 这是一次真实运行，不是基准测试，也不是教学效果或真人学习收益。
- 本机单用户；没有账户、租户隔离和内置限流。详见[公开边界](public-boundary.md)。

## 已验证运行（2026-11-14）

- 模型：`gpt-6.1-sol`；路由：`agent`；状态：`completed`。
- 运行：`ac4af32a-8e0f-4002-8a7c-54d67cf1d9a3`。
- 工具步骤：`inspect_page`/`inspect_media`（因资料 ID 未知返回 `material_not_found`）→ `list_materials` → `inspect_page` 第 10 页 → `inspect_media` 20–75 秒 → `deliver_explanation`。
- 证据：2 个已验证引用；第 4 步上下文含 3 张图片；字幕覆盖 20–75 秒；抽样帧为 33.750 秒和 61.250 秒。
- 回答摘录：“Post-norm 先把子层输出加回残差，再对和做 LayerNorm……Pre-norm 则先归一化再送进子层，子层输出才加回未归一化的 x_l，因此 x_l 有一条不经过 LayerNorm 的直通路径。”

原始运行记录、会话数据库和课程素材保留在忽略的 `data/`，公开文档只引用可复核路径和结论。

## 现场失败预案

- 没有配置模型：页面显示未配置模型，可先浏览已有会话和资料；不要伪造现场生成。
- 模型/网络不可用：明确说明是外部服务问题，展示已保存的真实运行记录，并标明是历史回放。
- 本机代理异常：如果 `NO_PROXY` 含 `[::1]` 等非法 URL 规则，httpx 可能无法发起请求；演示前改为 `NO_PROXY=localhost,127.0.0.1`。
- 缺少课程素材：使用[离线合成演示](demo-offline.md)，或上传自己的真实 PDF/视频；不要用合成幻灯片替代真实课程视频。
- 缺少 ffmpeg：只能浏览文字/PDF，无法生成视频回放和抽样帧；先安装 `ffmpeg`。
- 端口被占用：换 `--port`，不要结束已有服务。

## 不要做的事

- 不要把合成幻灯片、预写动画说成真实课程视频。
- 不要把课程视频、讲义或 ASR 全文提交到公开仓库。
- 不要把一次工具调用成功说成回答一定正确，也不要把开发复核说成真人学习收益。
- 不要把“抽样帧 + 字幕”说成逐帧观看或直接理解音频。