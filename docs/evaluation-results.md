# 评测结果（本机真跑账目）

这份文件是**历次真跑的账目**：每次跑用了什么 suite、花了多少、结果如何。当前口径与指标定义
看[统一评测](evaluation.md)与[多模态评测](multimodal-eval.md)；检索指标的口径变更看
[检索口径沿革](retrieval-history.md)。

第一节是 2026-11-01 的路由试点，第二节起是后续的能力评测全量真跑。

这些是本地单次试点，不是基准测试，也不是教学效果研究。内容复核由开发 Agent 按固定 rubric 完成，不是独立人工评审；Token 是主模型 usage 合计，没有价格表，不换算金额。原始数据、回答和数据库在忽略的 `data/evaluations/`，不随仓库分发。

## 自动路由 vs 始终 Agent

同一批 3 个 Case、各 2 回合，共 6 回合；模型与网关相同，温度等由网关默认。

| 配置 | 交付 | 证据检查 | 工具调用/失败 | 主模型 Token | P50 / P95 | 内容复核 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 始终 Agent | 6/6 | 1/1 | 11/0 | 59,467 | 28.1s / 56.4s | 6/6 |
| 自动路由 | 6/6 | 1/1 | 7/0 | 44,152 | 22.8s / 50.0s | 6/6 |

自动路由在这组试点里减少约 26% 主模型 Token、4 次工具调用，交付、证据和内容复核没有下降。

Case 与路由：

- `approximation`：首轮 `direct`，追问 `agent`；
- `retrieve_runtime`：首轮 `fixed_rag`，追问 `agent`；
- `prenorm`：首轮 `agent`，追问 `agent`。

### 规则迭代

第一版路由把追问也降级为直答，Token 降到 25,695（约 −57%），但开发复核发现 `retrieve_runtime` 的追问把两个百分比说成“分母相同”，内容复核变为 5/6。因此第二版规定：**存在已完成历史回合时，追问一律回到 Agent**；只有首轮独立问题允许直答或固定检索。第二版仍减少约 26% Token，但内容恢复到 6/6。

这是一次小样本开发复核，不是统计显著性结论；它说明“省成本”必须和内容复核一起看。

## 多模态资料：Agent 与基线的差异

同一批 3 个多模态 Case、各 2 回合，共 6 回合。`direct`/`fixed_rag` 都是纯文本流程，没有查看页面图像或视频帧；`learning-agent` 可以按需调用 `inspect_page` / `inspect_media`。

| 配置 | 交付 | 证据检查通过 | 主模型 Token | P50 / P95 | 内容复核 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无检索直答 | 6/6 | 0/3 | 5,307 | 9.5s / 38.0s | 3/6 |
| 固定 RAG | 6/6 | 0/3 | 8,030 | 13.3s / 23.3s | 3/6 |
| learning-agent | 6/6 | 3/3 | 57,127 | 27.2s / 47.3s | 6/6 |

证据检查只说明模型输入里是否包含标注的图像/字幕位置，不代表讲解一定正确。开发复核中，三条“必须实际看图/看帧”的首轮解释（`prenorm`、`rms_notation`、`video_joint`）两组基线都没有完成对应观察，各失败一轮；三条追问的数学与概念回答正确，所以两组基线各 3/6。Agent 六轮都通过了对应观察和内容条件。复核来自开发 Agent，不是独立人工评审；这张表用于说明为什么视觉/时间问题应路由到 Agent，而不是用来宣称 Agent 在所有任务上更好。

## 纯文本基线（路由动机）

在 `approximation`、`retrieve_runtime`、`retrieve_residual` 三个纯文本 Case 上，固定 RAG 与 Agent 都是 6/6 交付，但 Agent 使用 76,990 Token，固定 RAG 使用 8,100 Token。该试点内容复核未完成，只作为“需要路由和成本控制”的动机，不作为质量结论。

## 能力评测全量真跑 v3（2026-11-18）

gpt-6-sol，56 turns 全量 + 全量复核。完整账目在 `data/evaluations/multimodal-agent-v3-full/`
（suite v3，commit `f12349d`，不随仓库分发）。这一节是**上一轮**记录，最新一轮（v4）见
[多模态评测](multimodal-eval.md#能力评测最近一次全量真跑-v42026-11-19)。

两轮真跑：

- **v2 首轮**（`multimodal-agent-v2-full/`，56 turns / 94 请求）：47 完成、7 预算耗尽。
  暴露出**取证链用例的系统性步数耗尽**：4 步全被 list→search→inspect 吃掉，
  没轮到交付；101 次工具调用 0 失败，矛盾在步数上限而非工具。
- **v3 重跑**（取证链用例 `max_steps` 4→6，其余不变）：**53 完成、1 预算耗尽**，
  6 条取证链用例全部恢复正常交付。

v3 数字（54.6 万 tokens）：

| 维度 | 结果 |
|---|---|
| turn 状态 | 47→**53 完成**，预算耗尽 7→**1**，run_failed 1 |
| 图像证据检查 | **9/9 通过** |
| runtime_delivery | 54/55 |
| **模型裁判复核** | **54/54 全部完成，0 严重错误** |
| 裁判评分（0–3） | 准确 2.93 / 回应卡点 2.96 / 清晰 2.96 / 反馈适应 3.00（27 条适用） |
| criteria 判 fail | 仅 2 条 |

多模态核心用例全部完成了"真的看了"的证据闭环：

| 用例 | 结果 | 证据 |
|---|---|---|
| `silent_video` | 完成，证据 pass | `inspect_media` 取了 00:08/00:15/00:22 三帧（落在 5–25 s 要求区间），答案把"画面观察"与"推测"分开写，明确标注无声无字幕 |
| `zh_chart` | 完成，证据 pass | 打开第 51 页图，答出陡峭方向抖动/平缓方向慢 |
| `matrix_shapes` | 完成，证据 pass | 纯图 PDF 页（文字层为空）看图答题 |
| `video_joint` | 完成，证据 pass | PDF 页图 + 视频帧混合证据 |
| `prenorm` | 完成，证据 pass | 第 10 页图 + 视频片段 |

本轮裁判抓到的两处内容缺口（证据门槛过了、内容没答全，正是复核的价值）：

1. `zh_chart`：答案讲了抖动与快慢失衡，但**没说出"Hessian 条件数过大"这个根源**——
   幻灯片上其实写了。看图证据 pass ≠ 从图里把关键概念读全。
2. `conflict_notes` 追问：补充了分母差异，但**没明确"课件作者为何采用该记法不可判定"**，
   这是 criteria 里"交代未知"的严格要求。

两个失败随后定位到根因并修复，真跑验证通过（2026-11-18 晚）：

- `retrieve_runtime`：根因是运行时对预算用尽后的任何工具调用一律拒绝，
  而模型在最后一步仍继续调工具。修复：**最后一步不再提供工具定义**，
  模型只能以正文交付（含"未完成部分"说明）。真跑验证：explain 以正文交付了
  实质答案（FLOPs 0.17% vs 25.5% 运行时间），followup 完成。
- `summary_recall`：两层问题。其一，**指令+完整工具 schema 固定开销约 14.2k
  单元从不参与预算削减**，12k 预算下摘要永远没机会触发——修复：消息削减到头后
  把工具定义紧缩为"名称+一句话描述"（省约 7k），且紧缩结果带回给调用方，
  否则实测体积与削减估算不一致；其二，用例预算 12k 对多步回合在数学上装不下
  （指令本身 6.3k），v4 升到 16k——历史 15.4k 仍超输入预算 12928，
  摘要依然是必要条件。真跑验证：explain 从摘要原话回忆出目标并诚实标注
  "依据摘要引文、未能重新核实"，followup 正确吸收纠正。
- **指令本身 6.3k 单元是新的最大固定开销**，是后续上下文工程的第一优先级。
  **已压缩（2026-11-19）**：INSTRUCTIONS+LEARNING_GUIDANCE 逐条对照重写，
  每一条行为约束（引用格式、诚实语义、学习状态规则）都保留，只合并表述——
  2384→1904 字，指令实测 6480→5076 单元（-22%），指令+完整工具 schema 固定开销
  14324→12920。新增守护测试 `test_instructions_stay_within_fixed_overhead_budget`
  （上限 5500），防止指令无序回涨；质量影响由 learning-core-v4 全量重跑验证。

工程口径教训（已写进使用方式）：`--max-turns` 是全实验 turn 预算、
`limits.main_model` 是全实验请求预算，全量跑需显式给足；
跑完后全量复核用 `regrade --judge-budget 60 --judge --limit 60`
（扩展操作以 `limits-extension` 记录进账目，不手改数据库）。

## 复现

自动路由与始终 Agent（同一实验，两个变体）：

```bash
uv run python -m learning_agent.evaluation run \
  --suite evals/learning-core-v1.json \
  --assets data/demo/assets.local.json \
  --variants evals/variants-auto.json \
  --case approximation --case retrieve_runtime --case prenorm \
  --output data/evaluations/routing-v2 --real

uv run python -m learning_agent.evaluation results \
  --experiment data/evaluations/routing-v2 \
  --output data/evaluations/routing-v2/results.md
```

多模态 Agent 与两组基线：

```bash
uv run python -m learning_agent.evaluation run \
  --suite evals/learning-core-v1.json \
  --assets data/demo/assets.local.json \
  --case prenorm --case rms_notation --case video_joint \
  --output data/evaluations/multimodal-agent --real

uv run python -m learning_agent.evaluation baseline \
  --suite evals/learning-core-v1.json \
  --assets data/demo/assets.local.json \
  --case prenorm --case rms_notation --case video_joint \
  --variants direct fixed_rag \
  --output data/evaluations/multimodal-baselines --real

uv run python -m learning_agent.evaluation results \
  --experiment data/evaluations/multimodal-agent \
  --baseline data/evaluations/multimodal-baselines \
  --output data/evaluations/multimodal-results.md
```

真实运行会产生模型费用；`data/demo/assets.local.json` 和课程资料只保留在本机。结果表只统计两边都存在的回合；内容复核未完成的回合不当作通过。

当期工程回归为 189 项通过、2 项按条件跳过（当期口径；现行数字见 README）；CI 同时运行公开检查、pytest、ruff、评测完整性检查和 Docker build。

相关设计与边界见[自动路由](routing.md)和[统一评测](evaluation.md)。
