# 多模态评测：覆盖范围、两个标注缺陷与分模态指标

更新于：2026-11-24 10:26（Asia/Shanghai，UTC+08:00）。这份文档回答一个问题：**这个项目的多模态能力到底被测了多少，
哪些是真测的、哪些只是看起来测了。**

**本文件只写当前口径（v5）的数字。** 旧版本的指标与变更原因在[检索口径沿革](retrieval-history.md)。

## 一句话结论

当前语料 7 讲 11 份材料、942 块、92 题（83 道可评分），向量 `bge-m3` + 精排
`bge-reranker-v2-m3-int8`，产品路径 Hit@5 是 **BM25 0.40 → 混合 0.639 → 加精排 0.807**。

覆盖面上还有两块短板，都不是"分数不够高"，而是**评测本身还没到位**：

- 视频占 740/942 块（78.6%），但分块 40–45 秒，定不了"10 秒内"的位置；
- 英文三讲（第四、九、十讲）走本地 ASR，没有断句与术语修正，转写质量成了这几讲的上限。

16 道漏检的构成很具体：跨讲对照 4 道全灭（`targets_mode=all` 要求两份证据同进前 5）、
英文三讲 7 道（第四讲 2、第九讲 2、第十讲 3）、讲义 PDF 3 道、第一讲视频 2 道。
跨讲那 4 道是能力缺口，不是参数问题；其余多数是标注粒度、分块切断与转写质量，
精排把它们与"排序没排上来"分开了。逐条原因见下文。

下面先给构成，再给这几组数。

## 语料构成

评测语料在 `data/evaluations/multimodal-v1/`，11 份材料（只放被题目用到的材料）：

| 资产 | 格式 | 规模 | 可文字检索 | 说明 |
|---|---|---|---|---|
| cs336-architecture.pdf | pdf | 68 页 / 69 chunk | 是 | 英文讲义，含架构图页 |
| optimization.pdf | pdf | 119 页 / 131 chunk | 是 | 英文讲义，含损失面示意图 |
| cs336-lecture.mp4 | video | 87 分钟 / 121 chunk | 是（本地 ASR 英文） | 整讲原片，1876 条字幕段 |
| cs336-lecture4.mp4 | video | **86 分钟 / 122 chunk** | 是（本地 ASR 英文） | attention alternatives，1592 段 |
| cs336-lecture9.mp4 | video | **78 分钟 / 108 chunk** | 是（本地 ASR 英文） | scaling laws，1586 段 |
| cs336-lecture10.mp4 | video | **86 分钟 / 123 chunk** | 是（本地 ASR 英文） | inference，1093 段 |
| ntu-lecture.mp4 | video | **102 分钟 / 144 chunk** | 是（官方中文字幕） | 整讲原片，2608 条字幕段 |
| hungyi-flash-attention.mp4 | video | **50 分钟 / 69 chunk** | 是（官方中文字幕） | Flash Attention，1389 段 |
| hungyi-kv-cache.mp4 | video | **39 分钟 / 53 chunk** | 是（官方中文字幕） | KV Cache，1124 段 |
| worksheet.pdf | pdf | 1 页 / **0 chunk** | **否** | 纯图页，文字层为空 |
| study-notes.md | text | 2 chunk | 是 | 合成笔记，开头自述含故意写错的声明，供"不可信材料"用例；不是正常学习笔记 |

合计 **942 个分块，其中视频 740 块（78.6%）**，视频总量 8.8 小时（新增 5 讲占 5.6 小时）。三点值得注意：

1. **粒度差反过来了**：切片语料时期 PDF 分块是视频的 33 倍（200 : 6），看起来视频不可能有
   定位精度；换成整讲后视频占 740 / 942 = **78.6%**，反而变成"视频块把讲义页挤出去"
   ——同一个事实在幻灯片和讲稿里各出现一次，是这套语料最需要小心的性质。
   扩到 7 讲之后还多了一层：**同一门课的不同讲之间也互相挤**（MoE、KV cache、
   attention 都在两讲里出现），这是第十讲那 3 道漏检的直接原因。
2. **视频按 40–45 秒切块，切点会落在句子中间**。`video_sliding_window` 那条的漏检
   就是定义句被切成两半，含后半句的块开头是 "to a small region around…"，语义残缺。
3. **只有 `worksheet.pdf` 是"不可文字检索"的材料**（文字层为空、0 分块），
   它对应 `matrix_scan` 那条题并标了 `unretrievable`，单列不进分母——拿文字检索
   去命中纯图页是不可能的事，算成失败是用错误的标准判分。
   无声视频不在检索语料里：它在检索题上没有任何目标，放进去不会产生任何指标，
   只会在读语料清单时被误当成"图片能力被测过"的证据。它作为输入类型由能力用例
   （`learning-core-v4.json` 的 `silent_video`，经 `bindings.local.json` 绑定）覆盖。

## 分模态指标（v5：7 讲语料，92 题，向量 bge-m3 + 精排 v2-m3-int8）

| 模态 | 题数 | BM25 | 混合 | 混合 + 精排 |
|---|---|---|---|---|
| pdf | 21 | 0.524 | **0.714** | **0.857** |
| video（合计） | 61 | 0.361 | **0.623** | **0.787** |
| └ 中文问 × 英文讲稿（跨语言） | 28 | 0.036 | **0.429** | **0.679** |
| └ 中文问 × 中文字幕 | 29 | 0.724 | **0.897** | **1.000** |
| └ 跨讲对照（`all` 模式） | 4 | 0/4 | 0/4 | 0/4 |
| text | 1 | 0.00 | 0.00 | **1.00** |
| image | 1 | 不计分 | 不计分 | 不计分 |
| 合计（可评分 83） | 83 | 0.398 | **0.639** | **0.807** |

复核产物：`data/evaluations/multimodal-v1/bge-m3-report.json`（BM25 / 混合）与 `data/evaluations/multimodal-v1/bge-m3-rerank-v2-m3-int8-report.json`（混合 + 精排）。

精排那一列是**产品路径**（`Retriever.search` 默认 depth=5）跑出来的，不是拿更深的
候选池截前 5。两者差 1–2 题：下面的消融表用 depth=40 的候选池，基线是 0.631。

**这张表必须带模型名**：向量 `bge-m3`（1024 维）、精排 `BAAI/bge-reranker-v2-m3-int8`（0.57 GB）。
少一个，这张表就不是这组数。换模型差多少、为什么换，见[检索口径沿革](retrieval-history.md)；
当前配置的取舍理由见[检索对照](retrieval.md)。

**`text` 那一行不要当成"文本检索能力"。** 它只有 1 道题，材料 `study-notes.md` 是
合成笔记（2 个分块，开头自述"含故意写错的声明"，用于不可信材料用例）。这一列能说明的
只是"纯文本材料和别的材料走同一条检索链路"，说明不了文本检索本身的水平。
补真实笔记需要真实来源，编一份进去就等于给自己造题，所以这一格先空着，等有真实材料再加。

### 按材料

83 道可评分题，混合 vs 精排——**这张表比总分有用**，它说明总分数被哪些材料拉着：

| 材料 | 题数 | 混合 | 精排 | 精排后仍漏的题 |
|---|---|---|---|---|
| NTU 讲座（中文整讲） | 17 | 16 | **17** | 无 |
| CS336 讲义 PDF | 15 | 10 | **12** | runtime、double_norm、pdf_parallel_layers |
| CS336 第一讲视频 | 10 | 3 | **8** | video_sliding_window、video_aspect_ratio |
| optimization.pdf | 6 | 5 | **6** | 无 |
| CS336 第四讲（attn alternatives） | 6 | 4 | 4 | cs4_moe_parallel、cs4_ssm_downside |
| CS336 第九讲（scaling laws） | 6 | 3 | **4** | cs9_method3、cs9_isoflops |
| CS336 第十讲（inference） | 6 | 2 | **3** | cs10_batch_limit、cs10_speculative、cs10_kv_recipe |
| 李宏毅 Flash Attention | 6 | 5 | **6** | 无 |
| 李宏毅 KV Cache | 6 | 5 | **6** | 无 |
| 合成笔记 | 1 | 0 | **1** | 无 |
| 跨讲对照（all 模式） | 4 | 0 | **0** | 四道全漏 |
| **合计** | **83** | **53** | **67** | 16 题 |

两个反直觉的地方：

1. **中文讲稳定满分，英文三讲仍在爬**：李宏毅两讲 12/12；CS336 第四/九/十讲合计
   11/18（混合时 9/18）。这三讲英文是本地 ASR 出来的，没有标点断句、
   术语拼写还有错（"flotswaget" 应是 FLOPs）。逐对读原文的精排对干净字幕几乎全中，
   对转写质量差的稿子只补回 2 题——**这不是标注问题，是转写质量决定上限**。
2. **跨讲题全灭**是 `all` 模式的设计后果：它要求两个证据都在前 5，而两讲各 100 多块，
   凑齐的概率远低于单证据。这个 0 不是"检索全错"，而是"还做不到跨材料对照"。
   精排把单材料命中率提到 0.85（79 题中 67 题）之后**跨讲仍是 0/4**（下一节逐目标查过），
   而"把证据都送进候选池"也救不回来——瓶颈在排序决策，不在"单讲排序不够好"。

### 能不能靠后处理救回跨讲题：三条改法的实测

跨讲 4 题全灭之后，我先把两个假设当成真的去改，结果**都被数据否定**。
下表统一口径：候选深度 40，向量 `bge-m3`，精排 `bge-reranker-v2-m3-int8`：

| 配置 | 总 Hit@5 | MRR@5 | any Hit@5 | 跨讲 all |
|---|---|---|---|---|
| hybrid（现状，bge-m3） | 0.631 | 0.459 | 0.662 | 0/4 |
| hybrid + 每材料最多 3 条 | 0.583 | 0.447 | 0.613 | 0/4 |
| hybrid + 子查询融合 | 0.595 | 0.424 | 0.625 | 0/4 |
| **hybrid + 交叉编码器精排** | **0.798** | **0.674** | **0.838** | 0/4 |
| hybrid + 子查询融合 + 精排 | 0.786 | 0.671 | 0.825 | 0/4 |
| bm25 + 精排（不开向量） | 0.548 | 0.473 | 0.575 | 0/4 |

- **按材料限制候选条数**（假设"同讲冗余把别的讲挤掉了"）：每材料 3 条让指标掉 4.8pt，
  "前 5 被单一材料占满"的题从 26 道降到 0，跨讲仍 0/4——**占了位置并不是原因**。
- **子查询融合**（假设"一个查询里两个语义中心被平均掉"）：单独用掉到 0.595，
  叠上精排比纯精排低 1.2pt（0.786 vs 0.798），跨讲仍 0/4——**拆开查询没把缺的证据召回来**，
  还引入了噪音。逐子查询查过：连「KV cache 剪枝」都召不回原文写着 "KV Cache 的 pruning"
  的那块（`scripts/inspect_subquery.py`）。
- **精排是唯一确定有效的**：+16.7pt，any 类 0.838。而"只给 BM25 候选加精排"只有 0.548，
  说明它与向量召回各管一段，不能互相替代。

跨讲题到底缺什么，把每个目标摊在 120 名候选池里看（`scripts/inspect_targets.py`，已开精排）：

| 题 | 目标区间 | 精确命中（bge-m3 + 精排 v2-m3-int8，池深 120） |
|---|---|---|
| cross_serve_cost | cs9 [4462, 4505] | 第 1 名 ✓ |
| | cs10 [2539, 2579] | **池内无此区间**（cs10 只进池 2 条） |
| cross_flash_kv | hy_flash [873, 916] | **池内无此区间**（该材料进池 15 条，都不覆盖这段） |
| | hy_kv [181, 224] | **池内无此区间**（该材料进池 11 条） |
| cross_long_context | cs4 [2022, 2066] | 第 24 名 |
| | hy_kv [1059, 1103] | 第 12 名 |
| cross_kv_reduce | hy_kv [1410, 1452] | 第 6 名 |
| | cs10 [3840, 3884] | 第 29 名 |

**结论**：失败不只在召回，也不只在排序。分面召回把 8 份证据里的 7 份送进候选池
（每材料前 30 条）之后，前 5 仍然 0/4；融合权重从 0 扫到 0.9，也没有一档能同时容下两份证据。
真正的决策者是精排，而它在按"字面与问句的贴合度"排：目标块可能整块不含关键词
（hy_flash 全篇没有 "Flash Attention"），同材料里也可能有一块把结论说得更直接。
换更强的双塔（`qwen3-embedding:0.6b`、1024 维）之后，同一批块的精排名次几乎不动
（cs10 第 42、hy_flash 第 62），说明瓶颈不在召回侧。

"差一秒的相邻块"不能算命中：`cross_kv_reduce` 里紧贴目标区间的那条讲的是 attention sink，
不是 KV cache 剪枝；允许相邻块计入，等于用错的证据蒙对。要改善只能改召回侧
（按讲分别召回再合并、或扩大精排候选池），不是放宽判定。

进一步的分层诊断（`scripts/inspect_within_material.py` + `inspect_ce_neighbors.py`）把位置摊得更细：
把每份材料**单独**排序，目标块在材料内的 RRF 名次是 **1–67 名**（hy_flash 在 69 块里排 67），
过一遍精排后是 **1–62 名**（cs4 升到第 1，hy_flash 仍第 62）。这两个数不能互相替代：
cs4 那份材料内 RRF 第 4、精排第 1，进了跨材料池（池深 120）就掉到第 24——材料内的好名次会被全局分数合并抹平，
这也是"分面召回必须按材料内名次交错合并"的原因。

四条路径（扩候选池 / 分材料召回 / 换向量模型 / 改分块）已全部实测完，
结论与逐条数字写在[跨材料检索方案](cross-material-retrieval-plan.md)：
**0/4 没能救回来**。换精排已采纳，当前检索口径见[检索对照](retrieval.md)。

这两条被否定的改法留在 `scripts/retrieval_ablation.py` 里（`--quota` 可复现），
不是为了展示"试过多少种"，而是防止以后有人再把它们当新想法重来一遍。

> 早先两版评测集（12 题、17 题）与它们的分模态表、以及"换成两讲整讲后数字掉一档"的过程，
> 全部记在[检索口径沿革](retrieval-history.md)。那些数字和今天不可比（语料、向量模型都换过），
> 所以不放这里。

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

要做到独立图片支持，需要改三处：上传白名单、入库与缩略图、以及无文字层时的索引策略。
第三处的 PDF 扫描页版本已经补上（见[扫描页的文字识别](background-jobs.md#扫描页的文字识别)），
独立图片文件仍不支持。这些是功能改动，不是补评测，因此没有混进上面的评测结论。

## 复现方式

```bash
uv sync --extra embed
export LA_EMBEDDING_CACHE_DIR=~/.cache/la-models

uv run python scripts/check_annotation_targets.py --queries evals/retrieval-v5.json
uv run python scripts/retrieval_report.py \
    --corpus data/evaluations/multimodal-v1/corpus/workspace.db \
    --queries evals/retrieval-v5.json \
    --json data/evaluations/multimodal-v1/report.json
```

重建整讲语料（可续跑，已入库的材料会跳过；ASR 约 6 分钟）：

```bash
uv run python scripts/build_lecture_corpus.py --force   # --force 才会清空重来
uv run python scripts/prepare_multimodal_suite.py       # 能力用例的绑定与 suite
```

无声视频的生成方式（`cs336-prenorm-silent.mp4`，已入库，一般不用重跑）：

```bash
ffmpeg -ss 20 -t 30 -i <原视频> -an -c:v libx264 -crf 24 <输出>.mp4
```

能力用例（`learning-core-v4.json`，28 cases / 56 turns）需要真实模型调用，
用 `python -m learning_agent.evaluation` 跑，会产生 LLM 费用，与检索评测分开执行。

> 文件名口径（2026-11-23 统一）：该文件历史上叫 `learning-core-v2.json`，但内部
> `version` 字段早已升到 4，两个编号空间容易混。现在文件名、`id`、`version` 统一为
> v4；历史评测产物里的 `suite_id: learning-core-v2` 是当时记录，保留未改。

## 能力评测最近一次全量真跑 v4（2026-11-19）

> 上一轮（v3，2026-11-18）的完整账目在[评测结果](evaluation-results.md#能力评测全量真跑-v32026-11-18)。
> 那一轮把取证链用例的 `max_steps` 从 4 提到 6，完成 53/56；下面是瘦身与精排融合落地后的重跑。

v4（suite version=4，commit `e8ef9d0`，gpt-6-sol，56 回合全量，
102 个主模型请求 ≈ 55 万 tokens）在两项改进落地后重跑：

- **完成 50/56**（v3 为 53/56）。差异回合逐个定性：
  - `matrix_shapes:explain` interrupted——第 2 步请求 90 秒网关超时
    （`remote_result_unknown`），纯网络偶然；同用例单独补跑 2/2 completed，
    证明用例本身不受指令瘦身影响。评测框架对 interrupted 无自动重试
    （request_id 绑定原 run，设计如此），补跑走独立目录、不混入主账本。
  - `conflict_notes` / `summary_recall`×2 budget_exhausted——都是 max_steps=4
    用例在第 4 步（无工具、只许正文）以正文交付了实质答案：矛盾指认带
    PDF 页码引用、摘要回忆诚实标注、followup 正确吸收纠正。行为变化：
    v3 里模型第 4 步调用 deliver_explanation（completed），v4 瘦身后的指令
    让模型更倾向直接正文（budget_exhausted 但答案在场）。
- **judge 全量复核 54/54：criteria 52/56 pass，与 v3 持平**。v3 的
  `conflict_notes:followup` fail（未说明不可判定）在 v4 修复；
  四维评分 accuracy 2.89 / 卡点 2.91 / 清晰 2.98 / 反馈 2.88（v3 为
  2.93/2.96/2.96/3.00），差异在逐回合内容不同的波动范围内。
- **发现并修复 judge 复核盲区**：`summary_recall:explain` 被判 serious_error
  （"断言目标原话缺乏依据"），但 trace 取证显示摘要 payload 里的
  `user_quote` 与模型引用**逐字一致**——摘要触发后早期对话被移出输入，
  judge 只看 conversation_history 看不到摘要，把如实引用误判成捏造。
  修复：judge 复核输入注入 `summary_evidence`（按 input_ref 的 summary_id
  从 trial workspace 读摘要条目），指令明确"引用摘要 user_quote 属于有据
  引用"；新增测试 `test_summary_evidence_exposes_quoted_context_to_the_judge`。
  **已闭环验证**（patch 目录 `multimodal-agent-v4-patch-summary5`，旧 key 恢复
  后补跑）：同用例重跑 → summary 触发、模型如实引用摘要条目（含 source_run_ids
  溯源）并主动声明"只能依据二手摘要确认"；修复后 judge 重评 explain 回合
  criteria pass、serious_error 转 False（followup 全 3 分 pass），复核记录中
  可见 judge 输入确实携带 `summary_evidence`。
## 已知局限

- 视频题 61 条，比过去厚得多，但 `video_cite_papers` 这类中文问英文字幕的题余量仍不大
  （命中常在 3–5 名），一次排序波动就会掉出去。
- 视频 chunk 粒度是 40–45 秒，无法支撑"定位到 10 秒内"的精度；
  若要做到，需要改切片策略并按词级时间戳对齐，会影响已有向量，尚未做。
- 无声视频是从现有视频裁的 30 秒，只能测"是否看图"，不能测长视频里的跨段推理。
- 独立图片材料仍未支持（见上文边界），相关能力只有 PDF 图页与视频帧两条通路。
- 跨讲对照 4 道全灭：评测要求两份证据同进前 5，当前检索做不到，这是能力缺口不是标注问题。
- 能力评测最近一次全量是 2026-11-19（v4），之后没有再跑；指令与检索又改过，数字已不完全对应当前代码。
