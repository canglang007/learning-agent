# 参与开发

这个项目还在早期，主要是本机单用户的学习工具。Issue 和 PR 都欢迎，但请先看下面的边界，能省很多来回。

## 先确认环境

```bash
git clone <repo>
cd learning-agent
uv sync --extra dev
cp .env.example .env      # 只填 LA_BASE_URL / LA_MODEL / LA_API_KEY
```

跑测试和静态检查（CI 用同样的命令）：

```bash
uv run pytest -q
uv run ruff check .
uv run python scripts/check_public.py
uv run python scripts/check_evaluation.py
```

没有模型 key 时测试也能跑：测试用脚本化的假 provider，不发真实请求。

> **用 Google Drive 等云盘同步本目录的注意**：`uv` 创建的 `.venv` 默认在 `.venv/bin/` 放指向本机 Python 的符号链接（symlink），云盘会把它们当"替身"报错或跳过。本机副本已把 `bin/python`、`bin/python3`、`bin/python3.13` 换为实体文件（硬链接），并在 `.venv/lib/` 硬链接了 `libpython3.13.dylib`，可正常同步。若在新机器上重新 `uv venv` 或删除重建 `.venv`，symlink 会回来；同步目录下不要重建环境，或重建后按同样方式把这三个入口替换为实体文件。`uv sync`（不删环境）不会重建这些入口。

## 改代码前值得知道的约定

- **不猜模型能力**。协议、视觉、流式、上下文窗口都按服务商实际验证的结果写，不根据模型名推断；不确定就留成显式配置项或明确报错。
- **不静默降级**。检索、转写、视觉做不到时要说清楚发生了什么，而不是悄悄换成更弱的结果。
- **不隐藏失败**。模型调用失败、工具失败、引用对不上，都要在返回里体现；不要用兜底文案掩盖。
- **数值口径不擅自改**。预算、阈值、精度相关的改动请单独说明理由和影响范围。
- **资料与产物进 `data/`，个人记录进 `notes/`**，两者都在 `.gitignore` 里；不要在提交里带真实课程素材、会话数据库或截图中的课程内容。

## 提交 PR

1. 一个 PR 做一件事；修 bug 附最小复现，加功能说明场景。
2. 行为变化要配测试。现有测试分在 `tests/`，按模块命名（`test_retrieval.py`、`test_isolation.py` 等）。
3. 改动涉及用户可见行为时，同步更新 README 或对应 `docs/`；文档写了但代码没做，比不写更糟。
4. 提交前跑一遍上面四条命令，全绿再发。

## 不打算接受的方向

- 把账号体系、云托管或付费基础设施变成"跑起来"的前提——本地最小路径必须保持无鉴权即可启动（账号是可选开关，不是前置条件）；
- 把某一家模型服务硬编码成唯一选择；
- 用"降低精度/跳过校验"换速度的优化，除非有单独的论证和开关。

## 安全问题

不要开公开 Issue，按 [SECURITY.md](SECURITY.md) 走私密报告渠道。
