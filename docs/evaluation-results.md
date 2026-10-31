# 评测结果（本机试点）

更新于：2026-11-01 00:24:57（Asia/Shanghai，UTC+08:00）。

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

相关设计与边界见[自动路由](routing.md)和[统一评测](evaluation.md)。
