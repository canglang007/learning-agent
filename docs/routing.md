# 自动路由：直答、固定检索与 Agent

更新于：2026-11-15 22:20:00（Asia/Shanghai，UTC+08:00）。

默认 `LA_ROUTE_POLICY=agent`，保持原来的完整 Agent 行为。设置为 `auto` 后，Runtime 会在提交回合时选择一个执行模式；路由是确定性的，不额外调用模型。

## 三种模式

| 模式 | 适用问题 | 执行方式 |
| --- | --- | --- |
| `direct` | 一般概念、计算或解释，没有资料引用 | 一次模型调用，不检索，不调用工具 |
| `fixed_rag` | 明确问资料内容，但没有页面/图/视频/时间线索 | 检索前 5 段，拼入一次模型请求，不调用工具 |
| `agent` | 用户选中了位置、提交了反馈、问题涉及 PDF 页面/图表/视频/字幕/时间，或路由不确定 | 现有 Agent 循环、工具、预算与恢复 |

## 路由顺序

1. `LA_ROUTE_POLICY` 不是 `auto` → `agent`。
2. 有 `focus` → `agent`。
3. 有 `feedback` → `agent`。
4. 已有完成的上一回合（本次是追问）→ `agent`。
5. 问题包含“第 N 页、PDF、页面、图、公式、视频、字幕、帧、秒、回放”等明确视觉/时间线索 → `agent`。
6. 问题提到当前空间**资料名里的课程标识**（如 `cs336-architecture.pdf` → `cs336`）→ `fixed_rag`。
7. 问题包含“资料、课件、讲义、笔记、材料、课程、论文、原文、出处、检索、搜索”等资料线索 → `fixed_rag`。
8. 其余 → `direct`。

课程标识不写死在代码里：路由时读取当前空间的资料名，取出其中形如 `cs336` / `stat20` 的标识词，再检查问题里是否作为独立词出现。换一门课只需有对应文件，不改路由代码；资料里没有该标识时不会凭空命中。

不确定时宁可回到 `agent`；反馈和选中位置永远不会被降级到简单模式。

## 运行与记录

- `direct` 和 `fixed_rag` 仍然经过同一个 Runtime：提交、SSE、取消、幂等和恢复都保留。
- 路由结果写入 `route.selected` 事件、上下文快照和归档；不会只存在于内存。
- `fixed_rag` 的检索片段同时写入本次输入清单，评测可以检查证据是否真的进入模型输入。
- 简单模式不更新学习状态；需要连续反馈、记忆或状态更新时，应走 `agent`。
- 检索失败时回退到 `agent`，不静默给出无依据的固定检索答案。

## 评测

同一批 Case 可以对比“始终 Agent”和“自动路由”：

```bash
# evals/variants-auto.json 包含 agent / auto 两个变体
uv run python -m learning_agent.evaluation run \
  --suite evals/learning-core-v1.json \
  --assets data/demo/assets.local.json \
  --variants evals/variants-auto.json \
  --output data/evaluations/routing-v1 --real
```

`results` 表增加 route 后，重点看：

- 简单文本 Case 的主模型 Token 是否下降；
- 多模态/证据 Case 的证据检查和内容复核是否退化；
- 路由错误的 Case 是否符合规则，是否需要加规则或回退到 Agent；
- P50/P95 是否因少一次工具循环而下降。

本机 6 回合试点中，自动路由把主模型 Token 从 59,467 降到 44,152（约 −26%），工具调用从 11 降到 7，内容复核仍为 6/6；详见[评测结果](evaluation-results.md)。

## 当前边界

- 规则是基于题面和输入信号的启发式，不是学习出来的分类器；不保证语义上完全正确。
- `fixed_rag` 是纯文本检索，不能看图；需要视觉证据时必须走 `agent`。
- 路由只决定执行模式，不替 Agent 决定教学行动。
- 目前是单用户本地实现，不包含按租户/用户配置路由策略。
