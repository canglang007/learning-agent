"""评测资产名 → 语料材料名的单一映射。

两个脚本共用它：``retrieval_report.py`` 把标注里的资产名解析成材料，
``check_annotation_targets.py`` 校验标注区间是否落在某个分块内。原来各写一份，
加一种材料得改两处；改漏一处不会报错——解析不到的资产只在 stderr 里警告一行，
对应的题就变成"静默漏检"，看起来像检索不行，其实是标注没接上。
"""

DEFAULT_ASSETS = {
    "cs_pdf": "cs336-architecture.pdf",
    "cs_video": "cs336-lecture.mp4",
    "cs4_video": "cs336-lecture4.mp4",
    "cs9_video": "cs336-lecture9.mp4",
    "cs10_video": "cs336-lecture10.mp4",
    "zh_video": "ntu-lecture.mp4",
    "hy_flash": "hungyi-flash-attention.mp4",
    "hy_kv": "hungyi-kv-cache.mp4",
    "optimization": "optimization.pdf",
    "worksheet": "worksheet.pdf",
    "notes": "study-notes.md",
}


def settings_for(work, args, **overrides):
    """评测脚本统一的 Settings 构造。

    必须有这一层：之前消融脚本里有一组对照自己写了 ``Settings(...)``，漏掉
    ``embedding_cache_dir``，精排模型就下到临时目录、一次 1 GB 以上，跑完还被
    删掉——两组对照都卡在下载上，看起来像"计算太慢"。缓存路径只能在一个地方给。

    向量来源也在这里统一：评测要能对比 ``local``（进程内 ONNX）与 ``ollama``
    （本机守护进程），只换模型不换代码路径，两档数字才可比。

    ``work`` 是被拷出来的临时语料目录，必须显式传——每个评测脚本都先复制一份
    语料再跑，忘了传就会写到真正的 ``data/`` 里去。
    """
    from learning_agent.config import Settings

    values = {
        "data_root": work,
        "retrieval_mode": "hybrid",
        "embedding_provider": getattr(args, "provider", "local"),
        "embedding_base_url": getattr(args, "base_url", "http://127.0.0.1:11434"),
        "embedding_cache_dir": getattr(args, "model_cache", None),
        "rerank_model": getattr(args, "rerank_model", ""),
    }
    if getattr(args, "model", None):
        values["embedding_model"] = args.model
    values.update(overrides)
    return Settings(_env_file=None, **values)
