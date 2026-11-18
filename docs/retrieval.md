# 检索：从 BM25 到混合检索

更新于：2026-11-18。这份文档记录检索方案的对照实验、错例和取舍理由，数字都可复现。

## 一句话结论

线上原先声明 `hybrid` 但向量通道 `disabled`，实际只跑 BM25。接上本地向量模型后，
同一套标注题上 **Hit@5 从 0.40 到 0.60，MRR@5 从 0.350 到 0.533**，代价是
1 核 836 MB 机器上进程常驻从 209 MB 涨到约 495 MB、查询多 0.08 s。

补齐视频与图像题之后（`evals/retrieval-v3.json`，17 题），这个提升幅度更大：
**0.43 → 0.79**。其中纯视频 5 题从 0.40 到 1.00，是混合检索真正不可替代的地方；
分模态拆解与两个被掩盖的标注缺陷见[多模态评测](multimodal-eval.md)。

## 为什么 BM25 不够

BM25 走字面匹配。中文按 bigram 切（无空格语言的标准做法），英文按词切。
问题出在**跨语言**：讲义是英文，提问是中文。

```
问：为什么正则化偏好简单模型？
BM25 命中：ntu-tool-use.mp4（无关）    ← 中文 bigram 在英文 PDF 里没有对应词
```

```
问：CS336归一化放在子层之前，为何不影响残差直通？
BM25 命中：空                          ← 全空，向量是唯一出路
```

这不是调参能修的：查询里根本没有出现在文档中的词。

## 对照实验

评测集 `evals/retrieval-v2.json`：12 题（10 条有标注目标、2 条负例），
标注到页或到时间段；`matrix_scan` 标注为「文字层为空、检索不可达」单列，不进分母。
语料 6 份材料、208 个 chunk（CS336 68 页 + optimization 119 页 + 两段视频 + 图片页 + 笔记）。

下面这张表用的是 v2（12 题）。v3 加了 5 条纯视频题、修掉两个标注缺陷后扩到 17 题，
总体变成 0.43 → 0.79；为避免两套数字混用，v3 的分模态结果只在
[多模态评测](multimodal-eval.md)里报，本文其余部分保持 v2，作为选型阶段的记录。

| 方案 | 参数量 | Hit@5 | MRR@5 | 常驻内存 | 结论 |
|---|---|---|---|---|---|
| lexical_baseline（旧） | — | 0.30 | 0.200 | — | 已废弃 |
| **bm25（线上此前实际状态）** | — | **0.40** | **0.350** | — | 跨语言题全空 |
| bge-small-zh-v1.5 | 24M | **0.30** | 0.300 | ~120 MB | **比 BM25 更差**，见下 |
| **multilingual-MiniLM-L12** | 118M | **0.60** | **0.533** | 495 MB | **采用** |
| bge-m3（历史参考） | 568M | 0.70 | 0.470 | >1.2 GB | 1 G 机器装不下 |

前四行是同一套语料与题目；bge-m3 那行来自更早一代评测集（course-retrieval-v1），
**不是同分布，只作量级参考**，不能拿来宣称"我们比它差 0.1"。

### 那个更差的方案值得单独说

先试的是 `bge-small-zh-v1.5`——24M 参数、95 MB、中文语义检索的常用选择。
结果 **0.30，低于 BM25 的 0.40**。

原因是它被训练成中文单语模型，对英文讲义的语义对齐很弱；dense 通道给出的
排名是噪音，RRF 融合时把 BM25 本来命中的题挤了出去：

| 题目 | bm25 | +bge-small-zh | +MiniLM |
|---|---|---|---|
| gradient（pre-norm 梯度稳定性） | **1** | — | — |
| rms_definition | 2 | 1 | 1 |

所以「小模型」不等于「省资源的等价替代」，**模型的语言覆盖必须匹配语料**。
这个否定结果直接决定了选型：要小，但必须多语言。

## 逐题变化（采用方案）

| 题目 | BM25 | 混合 | 说明 |
|---|---|---|---|
| pre_en | 1 | 1 | 保持 |
| rms_definition | 2 | **1** | 提升 |
| runtime | — | **3** | 「浮点运算量少但耗时多」新命中 |
| runtime_en | 1 | 1 | 保持 |
| gradient | **1** | — | **退步**：向量把正确的页挤出前 5 |
| regularization | — | **1** | 「为什么正则化偏好简单模型」从全空变第 1 |
| dot_product | — | **1** | 「内积与矩阵乘法的区别」从全空变第 1 |
| pre_zh | — | — | 仍失败，见下 |
| double_norm | — | — | 仍失败 |
| matrix_scan | — | — | 图片页，文字检索不可达（设计如此） |

净变化：修好 3 题、1 题退步、2 题仍失败。

**仍失败的 `pre_zh`**（"归一化放在子层之前为何不影响残差直通"，期望 p10–12）
是最典型的难题：中文口语化提问 + 英文 PPT 短句（"Set up LayerNorm so that it
doesn't affect the main residual signal"），118M 的多语言模型也拉不齐。
这条给了明确的下一步：**rerank**（用交叉编码器对 top20 重排），而不是继续换更大的双塔模型。

## 交叉编码器重排：补上了 pre_zh，也暴露了新的取舍（2026-11-18）

`LA_RERANK_MODEL=BAAI/bge-reranker-base`（568M 参数，ONNX 1.04 GB，MIT）对融合后
前 20 个候选逐对打分，只重排、不改变召回集合。同一语料、同一 17 题（`scripts/rerank_experiment.py`，
走产品 `search` 路径实测）：

| 模态 | hybrid Hit@5 | +rerank Hit@5 | hybrid MRR@5 | +rerank MRR@5 |
|---|---|---|---|---|
| pdf（8 题） | 0.62 | **0.88** | 0.542 | **0.688** |
| video（5 题） | **1.00** | 0.80 | **0.557** | 0.567 |
| text（1 题） | 1.00 | 1.00 | 1.000 | 1.000 |
| **合计（14 题）** | 0.79 | **0.86** | 0.580 | **0.667** |

逐题变化：**`pre_zh` 从不中 → 第 1**（交叉编码器逐对读 query 和原文，
补上了双塔模型拉不齐的跨语言对齐）；`gradient`、`double_norm` 从不中进前 5；
`video_opt_exception` 从第 4 升到第 1。代价同样真实：
**`video_cite_papers` 从第 5 掉出前 5**（中文问英文字幕，重排模型也没拉齐），
`runtime` 从第 3 掉出、`video_tool_example` 从 1 退到 2。

**结论：收益是真的，代价也是真的，所以做成显式开关（`LA_RERANK_MODEL`，默认关）。**
典型部署判断：问题以"中文问英文讲义"为主、且内存 ≥ 2 GB 时值得开；
视频字幕类问题占比高或只有 1 GB 内存时别开。重排的另一个代价是每次查询
多一轮逐对推理（1 核约 1–2 秒）。失败如实降级：`rerank_status=rerank_failed:*`，
不冒充重排发生过。

## 重排排序：纯替换 → 分数空间融合（2026-11-19）

逐题复盘推翻了"按查询语言分流"的直觉：受损 3 题（`runtime`、`video_cite_papers`、
`video_tool_example`）与受益题**都是中文查询**，退化与语言无关，而与
**查询-目标的词面重叠**有关——转述型查询（"耗时多""两篇工作是谁"）的目标
片段措辞与查询几乎不重叠，CE 置信度区分不开时被词面接近的干扰项挤掉。

三种排序策略对照（同一语料、同一 17 题、产品 `search` 路径，
`scripts/rerank_fusion_experiment.py`）：

| 策略 | Hit@5 | MRR@5 |
|---|---|---|
| hybrid RRF（无重排） | 0.786 | 0.580 |
| 纯 CE 替换（`rerank_ce_weight=1.0`） | 0.857 | 0.667 |
| **分数空间融合（`rerank_ce_weight=0.9`，默认）** | **1.000** | **0.705** |

融合公式：`score = w·sigmoid(CE) + (1-w)·归一化RRF`。CE 高分饱和（都 >0.99）
后概率差微乎其微，RRF 位次差相对变大，转述型查询的目标由 RRF 托底保住；
CE 分差显著时 sigmoid 接近线性，融合跟随 CE——`pre_zh` 仍第 1、
`video_opt_exception` 仍第 1，同时 `runtime` 保住第 3、`video_cite_papers`
保住第 5。逐题**没有任何用例比纯替换更差**。

位次空间的加权 RRF 融合试过（alpha 扫描 0.2–0.8）救不回受损题且 MRR 更低
（≤0.621）；分数空间是正确的作用域。权重在 0.85–0.9 达到 hit@5 全中——
如实声明：这是在 14 题可评分用例上选的参数，样本小，更大查询集上需复核。
单元测试 `test_fuse_ce_with_rrf_saturation_tiebreak_and_replace_mode`
锁定饱和托底与 `1.0` 纯替换两种行为。

## 实现要点

- **双通道 + RRF 融合**，不做加权。BM25 分数无上界、余弦相似度在 [-1,1]，
  量纲不同且没有可靠标定，加权等于拍脑袋；RRF 只用排名（`1/(60+rank)`），稳健。
- **索引与查询分离**：chunk 向量持久化在 `chunk_vectors`，`encoder` 字段是
  `["local", 模型名]` 的指纹。换模型/换地址自动失效并重建，不会混用两套向量。
- **降级必须说出来**：向量不可用时返回 `retrieval="bm25"` 且 `dense_status`
  给出原因（`disabled` / `index_incomplete` / `embedding_request_failed:*`），
  前端和 Agent 都看得到，不假装自己在做混合检索。
- **同页去重**：每页最多 2 条命中，避免 5 条结果全来自同一页。
- **中文分词**：bigram 滑窗（`归一化` → `归一`/`一化`），不引分词器依赖。

## 部署取舍：为什么是进程内 ONNX

向量模型有三条路：ollama、云端 API、进程内 ONNX。

- **云端 API**：当前网关只代理 LLM，`/embeddings` 直接 404（实测 `model_not_found`），
  没有可用的向量端点。
- **ollama**：常驻进程再加一份运行时，1 G 机器上启动与常驻都更重。
- **进程内 ONNX（采用）**：`fastembed` + onnxruntime，无外部服务、无网络依赖，
  模型只在用到时才加载（`LA_EMBEDDING_PROVIDER=local` 才 import onnxruntime），
  不开启向量检索的部署完全不受影响。

实测（Azure 1 vCPU / 836 MB，`LA_EMBEDDING_THREADS=2` 默认）：

| 指标 | 数值 |
|---|---|
| 模型加载（首次含下载） | 121 s |
| 索引 208 个 chunk | 16.9 s |
| 单条查询编码 | 0.08 s |
| 完整 search 调用 | 0.10 s |
| 进程 RSS / 峰值 | 495 MB / 505 MB |

内存是这台机器上最紧的一项：Web 进程本身约 209 MB，加载模型后约 495 MB，
峰值 505 MB，机器有 2 GB swap 兜底。若部署在 2 GB 以上机器则毫无压力。

**没有做 int8 二次量化**：fastembed 提供的是 fp16 权重，onnxruntime 的
`quantize_dynamic` 不接受 fp16 输入（`Type 'tensor(float16)' ... invalid`），
强行转 fp32 再量化会引入一层额外的数值改写。按"省计算只能来自算法本身"的原则，
这一步没有做，而是把选择放在模型规模上（118M 而非 568M）。

## 复现方式

```bash
uv sync --extra embed
export LA_EMBEDDING_CACHE_DIR=~/.cache/la-models   # 可选，固定缓存避免重复下载

uv run python scripts/retrieval_report.py \
    --corpus <含 chunks 的 workspace.db> \
    --queries evals/retrieval-v3.json \
    --json /tmp/retrieval.json
```

输出包含总体指标、分模态指标和逐题排名；`--model` 可换模型做同样的对照。
若语料里缺某个标注资产，脚本会先警告"相关题目必然漏检"，
避免把语料不匹配误读成模型不行。

重排对照（走产品 `search` 路径，首次含模型下载约 1 GB）：

```bash
uv run python scripts/rerank_experiment.py
```

输出 baseline 与 rerank 两个配置的总体/分模态指标和逐题变化，
原始数据留在 `data/evaluations/multimodal-v1/rerank-product-path.json`。

## 已知局限

- 10 条有效样本量很小，0.60 与 0.70 的差距在统计上没有意义；结论只在"向量通道有用"
  这个量级上成立。
- `gradient` 的退步说明融合策略还有空间：当前 RRF 对两路一视同仁，
  专业术语类查询其实更该信 BM25。
- 只覆盖中↔英与中↔中；未测日韩、公式主导的页面。
- 视频按时间段召回，标注粒度到分钟级，比页级更粗。
