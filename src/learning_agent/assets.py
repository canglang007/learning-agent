"""Validated atomic PNG caches; render children only write their attempt directory."""

import os
import tempfile
from contextvars import ContextVar
from pathlib import Path

from PIL import Image

staging_directory = ContextVar("render_staging_directory", default=None)


def png_size(path):
    try:
        if path.stat().st_size > 6_000_000:
            return None
        with Image.open(path) as image:
            if image.format != "PNG":
                return None
            size = image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
        return size
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def image_target(root, asset):
    canonical = root / asset
    stage = staging_directory.get()
    output = canonical if png_size(canonical) or stage is None else Path(stage) / asset
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def atomic_png(target, raw):
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".png")
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        if png_size(temporary) is None:
            raise ValueError("渲染结果不是完整可用的PNG")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
