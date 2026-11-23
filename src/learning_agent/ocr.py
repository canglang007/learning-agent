"""Optional OCR for PDF pages that carry no extractable text layer.

Only pages whose text layer is effectively empty are sent here, so ordinary
digital PDFs never pay this cost. The engine is imported lazily: without the
optional ``ocr`` extra the rest of the system keeps working and scanned pages
simply stay unreadable, which is reported rather than silently invented.
"""

import io
import threading

_LOCK = threading.Lock()
_ENGINE = None
_FAILED = None


class OcrError(RuntimeError):
    pass


def available():
    """True when the optional engine can be imported."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001 -- optional dependency boundary
        return False
    return True


def _engine():
    global _ENGINE, _FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _FAILED is not None:
        raise OcrError(_FAILED)
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:  # noqa: BLE001 -- optional dependency boundary
        _FAILED = "未安装 OCR 依赖，请运行 uv sync --extra ocr"
        raise OcrError(_FAILED) from None
    try:
        _ENGINE = RapidOCR()
    except Exception:  # noqa: BLE001 -- model download boundary; details may leak paths
        _FAILED = "OCR 引擎初始化失败，请检查模型文件与可用内存"
        raise OcrError(_FAILED) from None
    return _ENGINE


def _rows(result):
    """Normalise the several result shapes across engine versions."""
    if not result:
        return []
    rows = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box, text = item[0], item[1]
        score = item[2] if len(item) > 2 and isinstance(item[2], (int, float)) else 0.0
        try:
            top = min(float(point[1]) for point in box)
            left = min(float(point[0]) for point in box)
        except (TypeError, ValueError, IndexError):
            top, left = 0.0, 0.0
        label = str(text).strip()
        if label:
            rows.append({"text": label, "score": float(score), "top": top, "left": left})
    rows.sort(key=lambda row: (round(row["top"] / 8), row["left"]))
    return rows


def recognize(png):
    """Return text lines and their mean confidence for one rendered page."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:  # noqa: BLE001 -- core deps; a missing one is an install error
        raise OcrError("缺少图像处理依赖，请重新安装项目依赖") from None
    with Image.open(io.BytesIO(png)) as handle:
        pixels = np.array(handle.convert("RGB"))
    engine = _engine()
    try:
        with _LOCK:
            output = engine(pixels)
    except Exception:  # noqa: BLE001 -- native inference boundary; redact details
        raise OcrError("OCR 识别失败，请检查页面图像与可用内存") from None
    result = output[0] if isinstance(output, tuple) else output
    rows = _rows(result)
    text = "\n".join(row["text"] for row in rows)
    mean = round(sum(row["score"] for row in rows) / len(rows), 3) if rows else 0.0
    return {"text": text, "lines": len(rows), "mean_score": mean}


def recognize_page(store, space_id, material_id, page):
    """Recognise one page because a reader just asked for it, then cache the text.

    Two paths exist for the same engine on purpose. The background ``ocr`` job is
    the "make this document searchable" path: it walks every page and bumps the
    material revision once at the end, which is what regenerates chunks. This
    path is the opposite trade-off — it only makes the page in front of the
    reader readable *now*, so it leaves the revision untouched: no chunk is
    regenerated, no index is invalidated, no rebuild is queued. The page becomes
    readable, not searchable. Publishing stays an explicit action.

    A page that yields no text (a pure figure, a blank scan) is marked ``blank``
    rather than relabelled as recognised content, so an empty result is not
    mistaken for text and is not attempted again on the next read.
    """
    from learning_agent.pdf import render_page

    row = store.page(space_id, material_id, page)
    if (row.get("text_source") or "text") != "none":
        return row
    png, _, _ = render_page(store.root / "materials" / (material_id + ".pdf"), page)
    text = recognize(png)["text"]
    if not text.strip():
        store.mark_page_blank(space_id, material_id, page)
    else:
        store.save_ocr_page(space_id, material_id, page, text)
    return store.page(space_id, material_id, page)
