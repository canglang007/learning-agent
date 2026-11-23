"""Run with `python -m learning_agent.worker`; API startup never starts a worker."""

import argparse
import fcntl
import json
import os
import signal
import threading
import time

from learning_agent.config import Settings
from learning_agent.jobs import Jobs, JobStopped, configuration
from learning_agent.media import MediaError, _worker_lock_fd
from learning_agent.ocr import OcrError
from learning_agent.retrieval import EmbeddingError, Retriever
from learning_agent.store import Store


class Worker:
    def __init__(self, settings, retriever=None):
        self.settings = settings
        self.store = Store(settings.data_root)
        self.jobs = Jobs(self.store)
        self.retriever = retriever or Retriever(self.store, settings)
        self.closed = threading.Event()
        self.lock = None

    def __enter__(self):
        self.lock = (self.store.root / ".worker.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError(
                "该数据目录已有处理进程或尚未退出的媒体子进程；请勿重复启动"
            ) from None
        try:
            self.jobs.recover()
            self.migrate_legacy()
            self.heartbeat()
            self.thread = threading.Thread(target=self.keep_alive, daemon=True)
            self.thread.start()
            # SIGKILL cannot run cleanup. Children retain the same lock until they exit,
            # preventing a replacement worker from writing their temporary files concurrently.
            self.fd_token = _worker_lock_fd.set(self.lock.fileno())
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def heartbeat(self):
        with self.store.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO worker_status VALUES (1,?,?)", (os.getpid(), time.time())
            )

    def keep_alive(self):
        while not self.closed.wait(2):
            self.heartbeat()

    def __exit__(self, *args):
        self.closed.set()
        self.thread.join()
        with self.store.connect() as db:
            db.execute("DELETE FROM worker_status WHERE id=1")
        self.lock.close()
        _worker_lock_fd.reset(self.fd_token)
        self.lock = None

    def migrate_legacy(self):
        # Old BackgroundTasks had no durable results. Only reclaim unowned in-flight work.
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT m.id,m.space_id,media.state,media.transcript_state FROM materials m JOIN media ON media.material_id=m.id"
            ).fetchall()
        for row in rows:
            for kind, pending in [
                ("preview", row["state"] == "processing"),
                ("transcribe", row["transcript_state"] == "running"),
            ]:
                with self.store.connect() as db:
                    exists = db.execute(
                        "SELECT 1 FROM jobs WHERE target=? AND kind=?", (row["id"], kind)
                    ).fetchone()
                if not pending or exists:
                    continue
                if kind == "transcribe" and self.settings.asr_provider != "local":
                    with self.store.connect() as db:
                        db.execute(
                            "UPDATE media SET transcript_state='failed',transcript_error='旧转写已中断；未自动重发云端请求，请明确重试' WHERE material_id=?",
                            (row["id"],),
                        )
                else:
                    self.jobs.enqueue(kind, row["space_id"], self.settings, row["id"])

    def run_once(self):
        if self.lock is None:
            raise RuntimeError("请在Worker上下文中执行任务，以持有数据目录锁")
        job = self.jobs.claim()
        if not job:
            return False
        try:
            if json.loads(job["config"]) != configuration(job["kind"], self.settings):
                self.jobs.action(job["id"], "pause")
                self.jobs.stop(job, "处理配置已变化，请恢复原配置或提交新任务")
                return True
            {
                "preview": self.preview,
                "transcribe": self.transcribe,
                "index": self.index,
                "ocr": self.ocr,
            }[job["kind"]](job)
        except JobStopped:
            self.jobs.stop(job)
        except (MediaError, EmbeddingError, OcrError) as error:
            self.jobs.stop(job, str(error))
        except Exception:  # noqa: BLE001 -- redact errors at the process/provider boundary
            # Never persist raw upstream errors, request headers or keys.
            self.jobs.stop(job, "处理失败；请检查文件、依赖及可用磁盘空间后重试")
        return True

    def drain(self):
        """Explicit driver for scripts/tests, with the same persistent pipeline as production."""
        while self.run_once():
            pass

    def preview(self, job):
        from learning_agent.media import prepare, probe, remux

        info = self.store.media(job["space_id"], job["material_id"])
        self.jobs.checkpoint(job, "preview")
        # force_transcode 来自客户端：服务端判定"能播"只说明容器+编码在理论上受支持，
        # 用户手上的浏览器未必解得了 AV1，这时必须给一份 h264 预览。
        if info.get("browser_playable") and not info.get("force_transcode"):
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self.jobs.check(job, db)
                metadata = json.loads(
                    db.execute(
                        "SELECT metadata FROM media WHERE material_id=?", (job["material_id"],)
                    ).fetchone()[0]
                )
                metadata["preview_path"] = info["original_path"]
                db.execute(
                    "UPDATE media SET metadata=?,state='ready',error=NULL WHERE material_id=?",
                    (json.dumps(metadata), job["material_id"]),
                )
                self.jobs.finish(job, db)
            return
        if info["state"] == "ready" and (self.store.root / info["preview_path"]).is_file():
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self.jobs.finish(job, db)
            return
        directory = self.store.root / "jobs" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "preview.mp4"
        if not output.exists():
            temporary = directory / "preview.partial.mp4"
            try:
                source = self.store.root / info["original_path"]
                # 只差容器的文件走 remux（秒级、不损画质），编码不兼容才重编码。
                if info.get("playback") == "remux":
                    remux(source, temporary, info)
                else:
                    prepare(source, temporary, info)
                self.validate_preview(
                    probe(temporary, max_seconds=max(14400, info["duration"] + 1)), info
                )
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            # Rename succeeded before DB commit: validate and adopt without transcoding again.
            self.validate_preview(probe(output, max_seconds=max(14400, info["duration"] + 1)), info)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.jobs.check(job, db)
            metadata = json.loads(
                db.execute(
                    "SELECT metadata FROM media WHERE material_id=?", (job["material_id"],)
                ).fetchone()[0]
            )
            metadata["preview_path"] = str(output.relative_to(self.store.root))
            db.execute(
                "UPDATE media SET metadata=?,state='ready',error=NULL WHERE material_id=?",
                (
                    json.dumps(metadata),
                    job["material_id"],
                ),
            )
            self.jobs.finish(job, db)

    @staticmethod
    def validate_preview(actual, expected):
        if (
            actual["has_video"] != expected["has_video"]
            or actual["has_audio"] != expected["has_audio"]
            or abs(actual["duration"] - expected["duration"]) > max(1, expected["duration"] * 0.01)
        ):
            raise MediaError("预览轨道或时长与原文件不符，请重试")

    def transcribe(self, job):
        from learning_agent.media import asr_parts, merge_transcripts, transcribe_part

        info = self.store.media(job["space_id"], job["material_id"])
        result = self.jobs.part(job, "transcript")
        if result is None:
            results = []
            parts = asr_parts(info["duration"], self.settings)
            directory = self.store.root / "jobs" / job["id"]
            for part in parts:
                self.jobs.checkpoint(
                    job,
                    "transcribe",
                    len(results),
                    len(parts),
                    seconds=part["start"],
                    duration=info["duration"],
                )
                key = "asr:" + str(part["index"])
                saved = self.jobs.part(job, key)
                if saved is None:

                    def before_remote():
                        self.jobs.checkpoint(job, "transcribe", remote=True)

                    saved = transcribe_part(
                        self.store.root / info["original_path"],
                        part,
                        self.settings,
                        directory,
                        before_remote,
                    )
                    saved["created_attempt"] = job["attempt"]
                    self.jobs.save_part(job, key, saved)
                results.append(saved)
                self.jobs.checkpoint(
                    job, "transcribe", len(results), len(parts), seconds=part["end"]
                )
            result = merge_transcripts(results, info["duration"])
            self.jobs.save_part(job, "transcript", result)
        self.jobs.checkpoint(job, "publish_transcript")
        self.store.save_segments(
            job["space_id"],
            job["material_id"],
            result["segments"],
            f"asr:{self.settings.asr_provider}:{self.settings.asr_model}",
            alignment_notes=result["alignment_notes"],
            job=job,
            allow_empty=True,
        )
        # New subtitles invalidate this material's chunks; rebuild without being
        # asked, otherwise the freshly transcribed lecture stays unsearchable.
        self.jobs.queue_index(job["space_id"], self.settings)

    def ocr(self, job):
        """Recognise pages that carry no text layer, one committed page at a time.

        Per-page results are written without touching the material revision, so a
        pause or crash resumes on the remaining pages. The single revision bump in
        ``finalize_ocr`` is what makes the retrieval index rebuild afterwards, and
        that rebuild is queued here rather than left to a manual click: recognised
        text that nobody can search is the side effect this job used to leave
        behind.
        """
        from learning_agent.ocr import recognize
        from learning_agent.pdf import render_page

        space, material = job["space_id"], job["material_id"]
        pending = self.store.ocr_pending_pages(space, material)
        total = len(pending)
        if total:
            path = self.store.root / "materials" / (material + ".pdf")
            for done, page in enumerate(pending):
                self.jobs.checkpoint(job, f"page-{page}", done, total)
                png, _, _ = render_page(path, page)
                text = recognize(png)["text"]
                if text.strip():
                    self.store.save_ocr_page(space, material, page, text)
                else:
                    # Remember the attempt: this page is a figure or a blank
                    # scan, and re-running it every time would be wasted work.
                    self.store.mark_page_blank(space, material, page)
            self.jobs.checkpoint(job, "publish_text", total, total)
        published = self.store.finalize_ocr(space, material)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.jobs.finish(job, db)
        if published:
            # Revision changed, so every chunk of this material will be rebuilt.
            # Queue that rebuild now instead of asking the reader to notice a
            # stale index and trigger it by hand.
            self.jobs.queue_index(space, self.settings)

    def index(self, job):
        def observe(stage, completed, total, batch, db=None):
            if stage == "before":
                self.jobs.checkpoint(
                    job,
                    "embedding",
                    completed,
                    total,
                    remote=self.settings.embedding_provider == "compatible",
                )
            else:
                self.jobs.save_part(job, "batch:" + str(completed), batch, db)
                db.execute(
                    "UPDATE jobs SET completed=?,total=? WHERE id=?", (completed, total, job["id"])
                )

        self.retriever.prepare(job["space_id"], observer=observe, raise_errors=True)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.jobs.finish(job, db)


def main():
    parser = argparse.ArgumentParser(description="本地持久任务处理进程（macOS/Linux）")
    parser.add_argument("--drain", action="store_true", help="处理完当前队列后退出，供验收脚本使用")
    args = parser.parse_args()

    def stop(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    try:
        with Worker(Settings()) as worker:
            if args.drain:
                worker.drain()
            else:
                print("处理进程已启动；Ctrl-C退出，已保存成果保留。", flush=True)
                while True:
                    if not worker.run_once():
                        time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
