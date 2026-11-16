# 评测闭环：跑出来的结果怎么回到开发

更新于：2026-11-16（Asia/Shanghai，UTC+08:00）。

[统一评测](evaluation.md) 和 [本机结果](evaluation-results.md) 讲评测模块本身。这篇只讲一件事：跑完评测之后，结果去了哪里，下次改代码怎么知道有没有改坏。

前四步其实一直是这么干的，第五步今天才补上。所以这篇更像一份补课记录，不是从头设计的方案。

```mermaid
flowchart LR
  A[任务集与 gold<br/>evals/learning-core-v1.json] --> B[真实执行<br/>不绕过 Runtime]
  B --> C[分层判定<br/>规则 / 内容复核 / 可选裁判]
  C --> D[冻结质量基线<br/>evals/quality-baseline.json]
  D --> E[改动：提示词 / 路由 / 上下文]
  E --> B
  B --> F[质量门禁 gate<br/>不调用模型，只读冻结报告]
  F -->|passed| G[更新基线]
  F -->|regressed| E
  F -->|incomparable / 复核不足| H[先补齐可比性]
```

## 五步都在干什么

1. 先把题出好。`evals/learning-core-v1.json` 里 28 组、56 回合，18 dev、10 holdout。gold 标了证据目标、状态期望和数值检查。holdout 留着，没拿来调参数。

2. 真跑，不抄近路。评测从正常 Runtime 提交，不直接生成答案塞进去。每次模型调用的输入清单（协议组、页图或帧、命中的证据、摘要、注入的记忆）跟着 operation 一起冻住。这样事后能分清楚：工具确实返回过，和这一次模型真的看到了，是两回事。

3. 分开判。规则 grader 看交付和证据位置，内容质量交给开发 Agent 按 rubric 复核，或者用模型裁判（要多花钱）。执行成功、证据命中、内容正确是三个字段，不揉成一个总分。

4. 把认可的那次结果冻起来，存成 `evals/quality-baseline.json`。里面只有指标、口径和来源，没有答案，也没有课程素材。

5. 改完东西再跑一次，`gate` 读两份冻好的 JSON 就能判：范围还能不能比、复核够不够、硬指标掉没掉。结论用退出码给出来，方便脚本接。

## 真改过的三次

这几件事都不是设计稿，是实际发生过、又回头验证过的。

| 发现了什么 | 凭什么这么说 | 改了什么 | 重跑之后 |
|---|---|---|---|
| 路由第一版省了 57% Token，但把追问也降级了，内容复核掉到 5/6。错在把两个百分比说成分母相同 | `docs/evaluation-results.md` | 规定有历史回合时，追问一律回 Agent | 回到 6/6，还是省了约 26% Token |
| 早期回答把 `inspect_page` 这类内部工具名直接说给学生听 | 真实网页验收 | 改学生侧提示词 | 重新验收过；原来的评测结果留着，没用新提示去覆盖旧证据 |
| 第一次工具调用拿文件名当资料 ID，白跑出一个 `material_not_found` | 5 步、2 次失败的真实运行 | 提示词里写清 ID 只能来自工具结果，错误文案也给出下一步 | 重跑一次：4 步、0 失败，引用和图片数量没变 |

还有一种情况值得记一笔。RRF 把 Hit@5 从 0.40 拉到 0.60，看着挺漂亮，但六对端到端样本的内容复核没有显示出差异。所以文档里只写"这套标注查询上召回提高"，没写成"RRF 让讲解普遍更好"。评测在这里的作用反而是拦住了一个过度结论。

## 门禁怎么判

```bash
# 生成或更新基线，本地跑，要有 data/
uv run python -m learning_agent.evaluation gate \
  --experiment data/evaluations/evaluation-pair-v1 \
  --snapshot evals/quality-baseline.json \
  --note '基线说明'

# 改完之后，先刷报告，再判
uv run python -m learning_agent.evaluation report data/evaluations/my-new-run
uv run python -m learning_agent.evaluation gate \
  --baseline evals/quality-baseline.json \
  --experiment data/evaluations/my-new-run
```

| 结论 | 退出码 | 什么时候出现 |
|---|---:|---|
| `passed` | 0 | 范围一致、复核够，交付、证据、内容都没掉 |
| `regressed` | 1 | 交付数、证据通过数或已复核内容通过数下降，或者重大错误增加 |
| `incomparable` | 2 | suite/rubric 版本、变体或回合集合对不上 |
| `insufficient_evidence` | 3 | 复核覆盖率不够，或者复核数比基线还少 |

几个地方我刻意写得狠一点。范围对不上、复核没做完，都直接判失败，不给一个看着好看的绿灯。一个条件从 pass 掉到 fail，不会因为另一个条件变好就被抹平，重大错误单独算。Token 和耗时打印出来给人看，但不参与质量判定，何况当前 provider 也不保证输出 token 上限，那个额度本来就不是钱数。

## CI 只管不花钱的那部分

CI 里跑 `scripts/check_evaluation.py`，查任务集结构（id、rubric 版本、split、turn、gold 有没有缺或重）、基线 schema 和指标是否合法、校准集和查询集是不是空的。不调模型，也不依赖被忽略的 `data/`。

质量门禁我没放进 CI。它要模型调用、课程素材，还要开发复核，每次 push 都跑一遍既不便宜也不可靠。放在本地显式执行更实在。

## 说之前要交代的

基线只有 12 回合、3 个 Case，复核是开发 Agent 做的，不是独立人工金标准。它能当回归信号用，撑不起显著性结论。

那 12 回合里有 4 个是在实验预算下以 `budget_exhausted` 交付的，算"已交付"，但不算完整成功。`evidence_total=2` 的意思是这套任务里只有 2 个回合标了位置或模态证据目标，别讲成 12 回合都验了看图。也没有真人学习收益数据，评分不写掌握画像。

还有一块真没做：线上反馈回流。`export-run` 能导出匿名统计，但把线上失败沉淀成评测 case 还得人工确认，目前没有这个机制。这是下一步，不是已经有的东西。
