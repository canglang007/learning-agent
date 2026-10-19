"""PDF text and lazy visual rendering, with physical page coordinates."""

import io
import math
import threading
from contextlib import closing
from functools import wraps

import pypdfium2 as pdfium


class PdfError(ValueError):
    pass


_LOCK = threading.RLock()


def serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # PDFium is not thread-safe; serialize calls even when run in workers.
        with _LOCK:
            return function(*args, **kwargs)

    return wrapped


@serialized
def extract_pages(raw: bytes):
    try:
        with pdfium.PdfDocument(raw) as document:
            if not 1 <= len(document) <= 200:
                raise PdfError("初版支持1至200页 PDF，请拆分较长文档。")
            pages = []
            for index in range(len(document)):
                with closing(document[index]) as page:
                    width, height = page.get_size()
                    if not all(math.isfinite(v) and 0 < v < 20000 for v in (width, height)):
                        raise PdfError("PDF 页面尺寸无效。")
                    with closing(page.get_textpage()) as textpage:
                        text = textpage.get_text_bounded().replace("\r\n", "\n").replace("\r", "\n")
                    if len(text) > 150_000:
                        raise PdfError("单页文字过多，请检查文件。")
                    pages.append(
                        {"page": index + 1, "text": text, "width": width, "height": height}
                    )
            return pages
    except PdfError:
        raise
    except (pdfium.PdfiumError, OSError, ValueError):
        raise PdfError("无法读取 PDF，请确认文件未损坏、未加密且格式有效。") from None


@serialized
def render_page(path, page_number, region=None):
    """Region is top-left normalized [left, top, right, bottom]."""
    try:
        with pdfium.PdfDocument(path) as document, closing(document[page_number - 1]) as page:
            width, height = page.get_size()
            scale = min(2.0, 1800 / max(width, height))
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        if region:
            left, top, right, bottom = region
            image = image.crop(
                (
                    int(left * image.width),
                    int(top * image.height),
                    math.ceil(right * image.width),
                    math.ceil(bottom * image.height),
                )
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), image.width, image.height
    except (pdfium.PdfiumError, OSError, ValueError, IndexError):
        raise PdfError("PDF 页面渲染失败。") from None
