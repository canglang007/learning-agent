"""Offset-confirmed uploads. A per-session OS lock spans streaming, never a DB transaction."""

import asyncio
import fcntl
import os
from contextlib import contextmanager
from uuid import uuid4

from fastapi import HTTPException

from learning_agent.media import MEDIA_SUFFIXES


async def disk_call(function, *args):
    # Finish an in-flight write/move before releasing the file lock on request cancellation.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class Uploads:
    chunk_size = 8 * 1024**2

    def __init__(self, store, jobs, settings):
        self.store, self.jobs, self.settings = store, jobs, settings
        (store.root / "uploads").mkdir(exist_ok=True)
        with store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY, material_id TEXT NOT NULL, space_id TEXT NOT NULL,
                filename TEXT NOT NULL, suffix TEXT NOT NULL, size INTEGER NOT NULL,
                received INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'uploading',
                last_modified REAL, job_id TEXT)""")

    def get(self, identifier):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise LookupError("上传会话不存在")
        return {**dict(row), "chunk_size": self.chunk_size}

    def create(self, space, filename, size, last_modified=None):
        from pathlib import Path

        self.store.require_space(space)
        suffix = Path(filename).suffix.lower()
        if suffix not in MEDIA_SUFFIXES:
            raise HTTPException(422, "分块上传用于音视频文件")
        if not 0 < size <= self.settings.media_max_bytes:
            raise HTTPException(413, f"媒体文件须为1到{self.settings.media_max_bytes}字节")
        identifier, material = str(uuid4()), str(uuid4())
        directory = self.store.root / "uploads" / identifier
        directory.mkdir()
        (directory / "upload.part").touch()
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO uploads(id,material_id,space_id,filename,suffix,size,last_modified) VALUES (?,?,?,?,?,?,?)",
                (identifier, material, space, filename[:200], suffix, size, last_modified),
            )
        return self.get(identifier)

    @contextmanager
    def locked(self, identifier):
        self.get(identifier)  # Validate before using the opaque ID as a path.
        with (self.store.root / "uploads" / identifier / "session.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise HTTPException(409, "该上传正在接收或完成，请先查询状态") from None
            yield self.get(identifier)

    async def append(self, identifier, offset, stream):
        with self.locked(identifier) as item:
            if item["state"] != "uploading":
                raise HTTPException(409, "上传不处于接收状态")
            if offset != item["received"]:
                raise HTTPException(
                    409, {"message": "上传偏移不匹配", "received": item["received"]}
                )
            path = self.store.root / "uploads" / identifier / "upload.part"
            count = 0
            try:
                if path.stat().st_size < offset:
                    raise HTTPException(409, "已确认的文件块不完整，请取消本次上传后重新开始")
                with path.open("r+b") as file:
                    # Drop an unacknowledged tail left by a disconnected request or crash.
                    file.truncate(offset)
                    file.seek(offset)
                    async for block in stream:
                        count += len(block)
                        if count > self.chunk_size or offset + count > item["size"]:
                            raise HTTPException(413, "文件块超过协商大小或声明的文件总长度")
                        await disk_call(file.write, block)
                    if not count:
                        raise HTTPException(422, "文件块不能为空")
                    await disk_call(file.flush)
                    await disk_call(os.fsync, file.fileno())
                with self.store.connect() as db:
                    db.execute(
                        "UPDATE uploads SET received=? WHERE id=?", (offset + count, identifier)
                    )
            except OSError:
                raise HTTPException(
                    507, "无法写入文件，请检查磁盘空间；已确认部分仍可继续"
                ) from None
        return self.get(identifier)

    async def complete(self, identifier):
        with self.locked(identifier) as item:
            if item["state"] == "completed":
                return self.result(item)
            if item["state"] == "cancelled" or item["received"] != item["size"]:
                raise HTTPException(409, "请先完成所有文件块，或重新创建上传")
            with self.store.connect() as db:
                db.execute("UPDATE uploads SET state='finalizing' WHERE id=?", (identifier,))
            path = self.store.root / "uploads" / identifier / "upload.part"
            # The fixed ID and final original path make move-before-DB crashes reentrant.
            try:
                await disk_call(
                    self.store.add_media_file,
                    item["space_id"],
                    item["filename"],
                    path,
                    item["suffix"],
                    item["material_id"],
                    self.settings.media_max_seconds,
                )
            except OSError:
                raise HTTPException(507, "收录失败，请检查磁盘空间后重试完成上传") from None
            with self.store.connect() as db:
                existing = db.execute(
                    "SELECT id FROM jobs WHERE target=? AND kind='preview' ORDER BY created DESC LIMIT 1",
                    (item["material_id"],),
                ).fetchone()
            job_id = (
                existing[0]
                if existing
                else self.jobs.enqueue(
                    "preview", item["space_id"], self.settings, item["material_id"]
                )["id"]
            )
            with self.store.connect() as db:
                db.execute(
                    "UPDATE uploads SET state='completed',job_id=? WHERE id=?", (job_id, identifier)
                )
            return self.result(self.get(identifier))

    def result(self, item):
        media = self.store.media(item["space_id"], item["material_id"])
        return {**media, "id": item["material_id"], "job_id": item["job_id"]}

    def cancel(self, identifier):
        with self.locked(identifier) as item:
            with self.store.connect() as db:
                exists = db.execute(
                    "SELECT 1 FROM materials WHERE id=?", (item["material_id"],)
                ).fetchone()
            if exists:
                raise HTTPException(409, "资料已收录，请继续完成登记；取消上传不会删除资料")
            # Only our temporary copy is removed; the source on the user's computer is untouched.
            (self.store.root / "uploads" / identifier / "upload.part").unlink(missing_ok=True)
            original = (
                self.store.root / "materials" / item["material_id"] / ("original" + item["suffix"])
            )
            original.unlink(missing_ok=True)
            with self.store.connect() as db:
                db.execute("UPDATE uploads SET state='cancelled' WHERE id=?", (identifier,))
        return self.get(identifier)
