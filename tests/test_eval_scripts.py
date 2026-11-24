"""评测脚本的配置纪律：不许自己换掉向量模型。

这条测试的由来不是调试踩坑，是我违反纪律的一次改动：本机 ollama 没启动，
我把产品在用的 ``bge-m3`` 换成进程内的 384 维小模型，让评测先跑起来，
之后所有数字都是换过的被测对象测出来的。正确动作是 ``ollama serve``。
**环境没起来就把环境启动起来，不要换组件**——换组件等于换掉产品，
之后所有横向比较都失效。

所以这里把"评测脚本不得指定向量模型"钉成不变式：模型只能来自
``config.py`` 的默认值或命令行参数，脚本里不能出现字面量。
"""

import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _literal_assignments(path):
    """返回该文件里 (行号, 名字, 字面量) —— 形如 embedding_model="..." 的赋值。"""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.keyword):
            targets, value = [node.arg], node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            name = target.id if isinstance(target, ast.Name) else target
            if name in {"embedding_model", "embedding_provider"}:
                found.append((node.lineno, name, value.value))
    return found


def test_eval_scripts_never_pin_embedding_model():
    """评测脚本不许写死向量模型/来源，只能走 config 默认值或命令行参数。

    判定范围是"导入了 ``eval_assets`` 的脚本"——也就是所有会产出评测数字的脚本。
    语料构建脚本不在范围内：它们是离线造 fixture 的，显式指定 provider 是有意的。

    允许的唯一字面量是 ``embedding_provider="disabled"``：那是在构造 BM25 基线，
    本来就该明确关掉向量，不能靠默认值。
    """
    offenders = []
    for path in sorted(SCRIPTS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "eval_assets" not in text:
            continue
        offenders += [
            f"{path.name}:{line} {name}={value!r}"
            for line, name, value in _literal_assignments(path)
            if not (name == "embedding_provider" and value == "disabled")
        ]
    assert not offenders, (
        "这些地方把向量模型写死了，请改用 config 默认值或 --provider/--model 参数：\n"
        + "\n".join(offenders)
    )


def test_settings_for_inherits_product_defaults():
    """``settings_for`` 不传模型时必须落到产品默认值，并且带上模型缓存目录。"""
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from argparse import Namespace

    from eval_assets import settings_for

    from learning_agent.config import Settings

    args = Namespace(provider="ollama", base_url="http://127.0.0.1:11434", model_cache="/tmp/x")
    settings = settings_for(Path("/tmp/work"), args, rerank_model="")

    defaults = Settings.model_fields
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == defaults["embedding_model"].default
    # 缓存目录必须一路带下去：漏了它精排会把权重下到临时目录，一次 1 GB 以上。
    assert settings.embedding_cache_dir == "/tmp/x"
    assert settings.data_root == Path("/tmp/work")
