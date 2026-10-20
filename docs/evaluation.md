# 统一评测与失败分析

更新于：2026-10-20 14:59（Asia/Shanghai，UTC+08:00）。E1/E2/E3的本地首版已实现；[设计](evaluation-analysis-design.md)保留完整方案，本文说明实际接口与验收边界。

## 当前可用

- 正常Runtime提交、多轮反馈、独立Case会话、配置variant和repeat；不绕过执行器直接生成答案。
- 逐attempt实际输入清单：协议组/省略、页图/帧、已进入输入的工具证据、摘要与注入记忆ID。
- SQLite评测账目、幂等提交、显式continue、发出请求前的模型/摘要/裁判预算；unknown不自动重发。
- 规则检查与有理由的人工/开发Agent复核，独立可选模型裁判；保留每次评分修订。
- 位置命中、必要模态、状态与记忆注入检查，失败候选及可回访证据；没有证据时保留needs_review。
- 同任务配对对比、只读查看页、脱离讲解请求的重评、默认不含原文的普通运行导出。
- 三类旧报告适配：课程多轮、检索组件、RT3。缺少的输入清单/复核来源标unknown，不补造成绩。

教学质量按准确性、回应卡点、清晰度、反馈适应0–3分逐项解释；重大错误另列。条件判定为pass/fail/needs_review，分数不代替用户掌握程度。工具更多、答案更长不加分。

## 文件与运行

公开手写任务在evals/，代码在src/learning_agent/evaluation/。原始资料、转写、回答、截图、评测数据库与复核留在忽略的data/，研究过程留在notes/。

`evals/learning-core-v1.json`现有24组、48回合，16组dev/8组holdout，覆盖解释、图表、媒体、检索、反馈、状态、记忆与证据不足；**这不表示48回合已全部真实验收**。本轮真实试运行采用3组×2回合×2检索配置。其他任务已有结构与部分工程路径验证，holdout未用于本轮调优。

先按[课程资料说明](learning-evaluation.md)提供已有文件。准备器只复用已有转写，生成明确标为合成的worksheet.pdf与错误笔记，不下载课程、不转写、不调用讲解模型：

```bash
uv run python examples/prepare_course_demo.py
uv run python -m learning_agent.evaluation run \
  --suite evals/learning-core-v1.json --assets data/demo/assets.local.json \
  --output data/evaluations/my-run --case prenorm --plan
# 新目录；显式允许模型调用，会产生费用
uv run python -m learning_agent.evaluation run \
  --suite evals/learning-core-v1.json --assets data/demo/assets.local.json \
  --output data/evaluations/my-run --case prenorm --real
```

plan默认不触网；real准备向量索引时可能访问配置的向量服务，默认不启动云端ASR。端到端默认最多12回合、48次主模型、4次摘要、12次裁判。每次新operation在发送前预留，prepared恢复复用预留、returned不新增、unknown不退回；跨评测库与工作空间保守计账，不承诺远端恰好计费一次。当前provider不一定强制输出token上限，因此这些额度不是金额保证。

`--variants evals/variants.json`对比BM25和hybrid；向量服务与索引需实际就绪，报告记录真实retrieval模式。`--continue`只接续原计划，不替用户确认未知费用重试。中途修改初始化状态且无法确认结果时停止该Case，不能自动重复修改；使用新试验处理。新repeat是新会话，不能挑最好的一次报告。

```bash
# 组件检索：复用已有语料，12条查询；可能访问本地/云端向量服务，不调用讲解模型
uv run python -m learning_agent.evaluation retrieval data/evaluations/my-run --real
# 结构化裁判：会增加独立计费调用；出错不自动修复重发
uv run python -m learning_agent.evaluation regrade data/evaluations/my-run --judge --limit 12
# 导入带reviewer_kind/reviewer/criteria/scores/reasons/evidence_refs的JSON复核列表
uv run python -m learning_agent.evaluation regrade data/evaluations/my-run --reviews path/to/reviews.json
uv run python -m learning_agent.evaluation compare data/evaluations/my-run --variants bm25 rrf
uv run python -m learning_agent.evaluation report data/evaluations/my-run
```

同一组件试验不会自动补充耗尽的向量额度；新一轮完整对比使用新的输出目录。重复裁判仅通过显式`--retry-judge`，会增加费用且仍受原预算限制。规则重算、已有人工复核导入和report不会再次调用讲解模型。

输出有experiment.db、每Trial独立workspace、trace.json、report.json/Markdown、comparison.json及组件结果。普通运行导出无需标准答案：

```bash
uv run python -m learning_agent.evaluation export-run \
  --workspace data --run-id <运行ID> --output data/export.json
# --include-content才包含原回答、对话与工具证据；默认只输出运行统计和待复核结论。
```

默认导出不含原回答；分享前仍需检查自己写入的复核理由，系统不承诺自动消除所有个人信息。报告没有API key、认证头、原生reasoning或base64图片。

## 本地审查页

正常启动后访问`/evaluation`。选择实验 → 并列配置 → 打开某回合 → 看正式答案、评分、失败依据、每次操作和实际输入图片；PDF链接带页码，媒体链接带当前文件时间窗口。只读GET接口包括实验列表、汇总、Trial以及受控资料/图片读取；不接受任意文件路径，不提供网页上的付费运行按钮。

默认读取LA_DATA_ROOT/evaluations；可用LA_EVALUATION_ROOT或启动器--evaluation-root指定独立报告根目录。案例原始工作空间及资料需保留，拷贝report.json并不包含媒体。view.html使用textContent展示模型文本，避免将待评答案执行为HTML。

## 已完成的真实验收

使用既有CS336课程讲义/45分钟片段及CS231n正则化课件，gpt-6.1-sol，Chat Completions，Ollama bge-m3。真实记录留在data/evaluations/evaluation-pair-v1。

| 检查 | 实际结果 |
| --- | --- |
| 端到端 | 12回合全部交付：8个completed，4个达到预设max_steps=4后以budget_exhausted交付；后者单列，不视为无约束完整成功 |
| 调用 | 26次主模型attempt，BM25/RRF各13次；usage均已返回。一次本地工具因原生连接停滞变unknown，恢复复用了已返回模型结果 |
| 检索实际模式 | 主对比的BM25/RRF均按实际工具观察确认；RRF查询返回hybrid_rrf/ready |
| 开发复核 | 逐条核对页图、p12/p19文字与相关自动转写，核心概念/构造计算未发现重大错误；部分首答偏长，两份回答暴露内部工具名，已记录并修正面向学习者的提示 |
| 配对内容 | 六对核心条件均满足，未显示内容正确性的差异；不宣称RRF必然改善每类讲解 |
| 时间 | 一回合包含停滞与人工恢复等待，墙钟约4609.6秒，明确标不用于速度比较；其余样例也不足以证明普遍速度收益 |
| 检索组件 | 12查询，其中10个有标注目标、2个缺少可回答证据；标注目标Hit@5为BM25 0.40 / RRF 0.60，MRR@5为0.35 / 0.433。RRF十二次均实际生效；标注非穷尽，不能称完整召回率 |

组件指标与端到端结果分开：候选命中增加不自动意味着讲解变好；扫描型PDF没有文字索引时，需要看图或其他定位方式。目标位置没命中也可能存在有效替代来源。保留这些未决样例，不为了分数临时更换标注。

裁判单独做12条手写校准，含4正确/4错误/4证据不足，在本次运行中与开发标注12/12一致；另对真实12回合完成评分。两部分各12次裁判调用，独立记账。样例与参考标注由开发Agent编写，并非独立人类金标准；讲解与裁判同型号，有共同偏差风险。真实首轮裁判的早期快照缺少完整历史，因此部分feedback_adaptation为null；后续Collector已加入截至该run的最近8条会话记录，旧评分保留原覆盖限制。

## 失败分析与实现边界

既有逻辑step快照可被重试更新；现在main_model操作input_ref保存该次实际选择，能区分“工具返回过”和“该次真的附加了图/正文”。旧运行缺少此信息时仍标unknown。

失败按资料准备、provider/runtime、检索、工具选择、上下文、感知/解释、记忆状态、评分器分类，引用具体操作/位置。规则给症状，语义复核补依据；不能凭一次回答猜模型内部为何犯错。数值规则、题目或裁判错误也属于评测自身问题，不算Agent事实错误。

本轮原生SQLite线程栈显示并发open/close时等待互斥；本地数据库连接事务现由进程内可重入锁串行保护，事务中不await。并发读取/嵌套读取、完整回归和真实恢复已验证；这是本机短事务方案，多进程之间仍靠SQLite/WAL与既有任务所有权规则，不意味着具备多租户或分布式数据库能力。

后续可以扩大未运行任务、加入独立人工复核与真实学习者试用；本模块不测量用户长期学习收益，不把评分写成掌握画像。
