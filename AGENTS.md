# Learning Agent

更新于：2026-10-20 14:59（Asia/Shanghai，UTC+08:00）。

- 独立新项目，不迁移旧项目功能清单或默认复用旧代码。
- 核心是准确、清晰、针对困惑的讲解；检索、来源回访服务于理解。
- Agent 依据工具观察和用户反馈选择行动，不以固定 RAG 流程冒充自主决策。
- 依据实际代码与真实验收判断完成度；mock 测试不证明教学或模型效果。
- 原始资料、数据库与运行结果放在 data/；自用记录放在 notes/、experiments/，不提交。
- Git commit 消息使用中文。
- 新建或更新文档时，标注编写或更新时间，至少精确到分钟并注明时区；补写旧文档不得臆造历史时间。
- AGENTS.md 保持简短，不记录进度日志。公共文档区分已实现与计划能力。

- 本地入口为 `./start.sh`；统一评测入口为 `python -m learning_agent.evaluation`，使用与边界见 `docs/evaluation.md`。
