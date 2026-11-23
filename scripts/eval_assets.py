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
