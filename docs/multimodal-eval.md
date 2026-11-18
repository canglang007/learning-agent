# 多模态评测：覆盖范围、两个标注缺陷与分模态指标

更新于：2026-11-18。这份文档回答一个问题：**这个项目的多模态能力到底被测了多少，
哪些是真测的、哪些只是看起来测了。**

## 一句话结论

原先的评测**偏重 PDF 到近乎失真**：12 条检索题里有 8 条只指向 PDF，2 条虽标了视频但
只要 PDF 命中就算过；28 条能力用例里真正强制"必须看图/看帧"的只有 3 个 turn。
补上纯视频查询与图像证据门槛后，视频检索第一次有了独立数字：**BM25 Hit@5 0.40 →
混合检索 1.00**（5 条纯视频题）。

## 语料构成

评测语料在 `data/evaluations/multimodal-v1/`，7 份材料：

| 资产 | 格式 | 规模 | 可文字检索 | 说明 |
|---|---|---|---|---|
| cs336-architecture.pdf | pdf | 68 页 / 69 chunk | 是 | 英文讲义，含架构图页 |
| optimization.pdf | pdf | 119 页 / 131 chunk | 是 | 英文讲义，含损失面示意图 |
| cs336-prenorm.mp4 | video | 198 s / 3 chunk | 是（英文字幕） | 原始视频 353–551 s 片段 |
| ntu-tool-use.mp4 | video | 270 s / 3 chunk | 是（中文字幕） | 原始视频 3030–3300 s 片段 |
| **cs336-prenorm-silent.mp4** | video | 30 s / **0 chunk** | **否** | 去音轨，无字幕，只能看图 |
| worksheet.pdf | pdf | 1 页 / **0 chunk** | **否** | 纯图页，文字层为空 |
| study-notes.md | text | 2 chunk | 是 | 英文笔记 |

两点值得注意：

1. **PDF 的 chunk 数是视频的 33 倍**（200 : 6）。视频按 40–45 秒切块，一个讲解单元一块；
   PDF 按页切。这个粒度差不是 bug，但决定了视频不可能有页级的定位精度。
2. **两份"不可文字检索"的材料**是刻意保留的：它们把"看图的通路"单独隔离出来，
   否则图片能力永远会被文字检索的成功掩盖。

## 分模态指标（`evals/retrieval-v3.json`，17 题）

| 模态 | 题数 | BM25 Hit@5 | 混合 Hit@5 | MRR@5（混合） |
|---|---|---|---|---|
| pdf | 8 | 0.50 | 0.62 | 0.542 |
| **video** | **5** | **0.40** | **1.00** | **0.557** |
| text | 1 | 0.00 | 1.00 | 1.000 |
| image | 1 | 不计分 | 不计分 | — |
| 合计（有标注目标） | 14 | 0.43 | 0.79 | 0.580 |

`image` 那题（`matrix_scan`，问 worksheet 里的矩阵形状）标注为「文字层为空、
检索不可达」，单列不进分母——因为**让文字检索去命中纯图页是不可能的事**，
把它算成失败是拿错误的标准判分。它由能力用例（`inspect_page` + `modality: image`）覆盖。

### 视频那 5 题，逐题看

| 题目 | BM25 | 混合 | 证据内容 |
|---|---|---|---|
| video_opt_exception | — | 4 | 「唯一例外是 OPT 350M，训练时被孤立了」 |
| video_warmup | — | **1** | 「post-norm 不稳定，要配合 learning rate warm-up」 |
| video_cite_papers | — | 5 | 「Salazar and Yen，还有 Xiong 2020 那篇」 |
| video_tool_example | **1** | 1 | 「有个函数叫 Temperature，查某地某时温度」 |
| video_prompt_kind | **3** | 3 | 「这套说明叫 System Prompt，例子问高雄气温」 |

规律很清楚：**中文问中文字幕（下两条）BM25 就能命中；中文问英文字幕（上三条）
BM25 全空，只有向量能救。** 这正是跨语言检索在视频上的复现。

## 两个已修的标注缺陷

这两个缺陷有个共同点：**都是被 `any()` 语义掩盖的**——只要有一个目标命中，
另一个目标的错误永远不会暴露。

### 缺陷一：一个永远不可能成立的视频锚点

`retrieval-v2.json` 里 `runtime` 和 `runtime_en` 都标注了
`{"asset": "cs_video", "anchors": [859], "time_basis": "clip"}`。但：

- `cs336-prenorm.mp4` 的 `source_start_seconds = 353`、时长 198 秒 → 覆盖原始 353–551 秒
- `859` 不在这个区间内；换算成 clip 基准是 506 秒，也超过 198 秒上限

也就是说这个条件**在任何时间基准下都不成立**。而评分是 `any(...)`：PDF 第 16 页
命中了就算过，视频那条永假条件从未被触发，也从未被发现。

而且这条查询问的是"浮点运算量少但耗时多"，那段视频讲的是 pre-norm 与 post-norm
的对比，**内容本身也对不上**——不是标错时间，是把不存在的证据写进了标注。

修复：v3 移除这两个视频目标，并在 `annotation_evidence` 里记录原因。
同时加了一条回归测试，断言所有时间锚点必须落在资产时长内
（`tests/test_multimodal_eval.py::test_retrieval_v3_targets_stay_inside_asset_time_ranges`）。

### 缺陷二：三个没有资产、没有锚点的空壳用例

| 用例 | 原状 | 问题 |
|---|---|---|
| `silent_video` | 无 `focus`、`evidence_targets` 为空 | 问"这个没有字幕的视频"，但语料里**两个视频都有字幕**，没有对应资产 |
| `zh_chart` | 无 `focus`、`evidence_targets` 为空 | 问"训练损失下降但验证损失上升"，语料里**没有这张图**（搜 `train loss` / `validation loss` 全空） |
| `video_limit` | 有 `focus`、`evidence_targets` 为空 | 这条是**故意的**（考"不该编造"），但缺 `forbidden_claims`，编了也没记录 |

前两条的问题是同一个：`evidence_targets` 为空 → `grade()` 只检查"答案文字里有没有
某个概念"→ **模型完全没看画面、靠先验知识答对也能满分**。这恰好把多模态最核心的
能力（真的看了）测成了空白。

修复：

- **造了一份真的无声视频**：从 `cs336-prenorm.mp4` 的 20–50 秒裁出 30 秒、**去掉音轨**、
  以 `cs336-prenorm-silent.mp4` 入库。无音轨意味着 `has_audio=False`，
  转写任务直接跳过，因此**不可能有字幕**——语义上干净。
  它在语料里 `chunks = 0`，文字检索完全不可达，只能 `inspect_media` 看帧。
  用例要求图像帧落在 5–25 秒、`modality: image`：**没看图即失败**。
- **图表用例换成真实图表**：`zh_chart` 改为 `optimization.pdf` 第 51 页
  （SGD 在病态损失面上的抖动），`modality: image`、`pages: [51]`，
  给错页也判失败。
- `video_limit` 补 `forbidden_claims`，把"编造 4060 实测加速倍数"列入复核项。

## 覆盖边界：独立图片材料不支持

产品当前的输入格式是 `txt / md / pdf / audio / video`（服务端 `app.py` 的白名单与前端
`accept` 都是这几种）。**不能上传独立的 jpg/png**。

所以"图片能力"在本项目里的真实形态是：

- PDF 内嵌图页（`inspect_page`，含 `region` 裁切）
- 纯图 PDF 页（`worksheet.pdf`，文字层为空但能看图）
- 无字幕视频帧（`inspect_media` 的 `joint` 模式）

评测覆盖了后两种真实通路。**没有覆盖"独立图片文件"**，因为它不存在——
把不存在的通路写进评测，等于给自己发一张假证书。

要做到独立图片支持，需要改三处：上传白名单、入库与缩略图、以及无文字层时的索引策略
（OCR 或视觉描述）。这是新功能，不是补评测，因此没有混在这次改动里。

## 复现方式

```bash
uv sync --extra embed
export LA_EMBEDDING_CACHE_DIR=~/.cache/la-models

uv run python scripts/retrieval_report.py \
    --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
    --queries evals/retrieval-v3.json \
    --json data/evaluations/multimodal-v1/retrieval-v3.json
```

重建语料（只在需要重新生成无声视频时用）：

```bash
ffmpeg -ss 20 -t 30 -i <原视频> -an -c:v libx264 -crf 24 <输出>.mp4
uv run python scripts/prepare_multimodal_corpus.py
uv run python scripts/prepare_multimodal_suite.py
```

能力用例（`learning-core-v2.json`，28 cases / 56 turns）需要真实模型调用，
用 `python -m learning_agent.evaluation` 跑，会产生 LLM 费用，与检索评测分开执行。

## 已知局限

- 纯视频题只有 5 条，1 条失败就差 0.2；`video_cite_papers` 命中在第 5 名，
  说明跨语言视频检索余量不大，仍可能被挤出去。
- 视频 chunk 粒度是 40–45 秒，无法支撑"定位到 10 秒内"的精度；
  若要做到，需要改切片策略并按词级时间戳对齐，会影响已有向量，尚未做。
- 无声视频是从现有视频裁的 30 秒，只能测"是否看图"，不能测长视频里的跨段推理。
- 独立图片材料仍未支持（见上文边界），相关能力只有 PDF 图页与视频帧两条通路。
