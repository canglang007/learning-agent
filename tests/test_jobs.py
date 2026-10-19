import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import test_media
from fastapi.testclient import TestClient
from test_learning import Scripted

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.retrieval import EmbeddingError, Retriever
from learning_agent.store import Store
from learning_agent.worker import Worker


@pytest.fixture
def clip(tmp_path):
    return test_media.clip.__wrapped__(tmp_path)


def setup(tmp_path, **kwargs):
    settings = Settings(_env_file=None, data_root=tmp_path / "workspace", **kwargs)
    store = Store(settings.data_root)
    space = store.space("course")["id"]
    return settings, store, space, Jobs(store)


def test_concurrent_enqueue_claim_and_api_restart_do_not_reset_work(tmp_path):
    settings, store, space, jobs = setup(tmp_path, embedding_provider="ollama")
    store.add_material(space, "notes.txt", "attention")
    with ThreadPoolExecutor(2) as pool:
        ids = list(
            pool.map(
                lambda _: Jobs(Store(settings.data_root)).enqueue("index", space, settings)["id"],
                range(2),
            )
        )
    assert ids[0] == ids[1]
    with Worker(settings) as worker:
        job = jobs.claim()
        assert jobs.claim() is None
        create_app(settings, Scripted([]))
        assert jobs.get(job["id"])["state"] == "running"
        with pytest.raises(RuntimeError, match="已有处理进程"), Worker(settings):
            pass
        assert jobs.online()
        worker.jobs.action(job["id"], "cancel")
    assert not jobs.online()


def test_recovery_never_automatically_replays_unknown_remote_call(tmp_path):
    settings, _, space, jobs = setup(tmp_path, embedding_provider="compatible")
    item = jobs.enqueue("index", space, settings)
    with Worker(settings):
        job = jobs.claim()
        jobs.checkpoint(job, "embedding", remote=True)
    # Stopping/reopening API has no recovery side effect.
    create_app(settings, Scripted([]))
    assert jobs.get(item["id"])["state"] == "running"
    with Worker(settings):
        result = jobs.get(item["id"])
        assert result["state"] == "failed"
        assert "remote_result_unknown" in result["error"]
        assert jobs.claim() is None
        jobs.action(item["id"], "retry")
        assert jobs.claim()["attempt"] == 2


class Batches:
    key = "test-encoder"

    def __init__(self, hook=None):
        self.calls = []
        self.hook = hook

    def embed(self, texts):
        self.calls.append(len(texts))
        if self.hook:
            self.hook(len(self.calls))
        return [[1.0, 0.0] for _ in texts]


def test_index_pause_resume_and_failed_retry_reuse_committed_vectors(tmp_path):
    settings, store, space, jobs = setup(tmp_path, embedding_provider="ollama")
    store.add_material(space, "notes.txt", "attention " * 2200)
    job = jobs.enqueue("index", space, settings)
    embedder = Batches(lambda count: jobs.action(job["id"], "pause") if count == 1 else None)
    with Worker(settings, Retriever(store, settings, embedder)) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "paused"
    assert embedder.calls == [16]
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 16
    jobs.action(job["id"], "resume")

    def fail(_):
        raise EmbeddingError("offline")

    with Worker(settings, Retriever(store, settings, Batches(fail))) as worker:
        worker.drain()
    restarted = Retriever(Store(settings.data_root), settings, Batches())
    assert restarted.status(space)["state"] == "failed"
    assert restarted.status(space)["error"] == "offline"
    jobs.action(job["id"], "retry")
    with Worker(settings, restarted) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "succeeded"
    assert restarted.status(space)["dense_ready"]
    assert sum(restarted.embedder.calls) == restarted.status(space)["chunks"] - 16


def test_changed_configuration_pauses_instead_of_switching_models(tmp_path):
    settings, _, space, jobs = setup(tmp_path, embedding_provider="ollama")
    job = jobs.enqueue("index", space, settings)
    changed = settings.model_copy(update={"embedding_model": "other-model"})
    with Worker(changed) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "paused"
    assert "配置已变化" in jobs.get(job["id"])["error"]
    replacement = jobs.enqueue("index", space, changed)
    assert replacement["id"] != job["id"]
    assert jobs.get(job["id"])["state"] == "cancelled"


def test_preview_is_adopted_after_real_process_crash_before_database_commit(
    tmp_path, clip, monkeypatch
):
    settings, store, space, jobs = setup(tmp_path)
    media = store.add_media(space, "clip.mkv", clip, ".mkv")
    job = jobs.enqueue("preview", space, settings, media["material_id"])
    script = """
import os, sys
from pathlib import Path
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.worker import Worker
def crash(*args):
    os._exit(27)
Jobs.finish = crash
with Worker(Settings(_env_file=None, data_root=Path(sys.argv[1]))) as worker:
    worker.drain()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(settings.data_root)], timeout=30, check=False
    )
    assert result.returncode == 27
    output = settings.data_root / "jobs" / job["id"] / "preview.mp4"
    assert output.is_file()
    assert store.media(space, media["material_id"])["state"] == "processing"

    def no_reencode(*args):
        pytest.fail("Published artifact should be adopted without encoding again")

    monkeypatch.setattr("learning_agent.media.prepare", no_reencode)
    with Worker(settings) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "succeeded"
    assert store.media(space, media["material_id"])["preview_path"] == str(
        output.relative_to(store.root)
    )


def test_uploaded_captions_cancel_inflight_asr_and_preserve_old_until_publication(
    tmp_path, clip, monkeypatch
):
    # The real clip has no audio; mark a synthetic audio track in metadata for this race test.
    settings, store, space, jobs = setup(tmp_path, asr_provider="local")
    media = store.add_media(space, "clip.mkv", clip, ".mkv")
    identifier = media["material_id"]
    with store.connect() as db:
        metadata = json.loads(
            db.execute("SELECT metadata FROM media WHERE material_id=?", (identifier,)).fetchone()[
                0
            ]
        )
        metadata["has_audio"] = True
        db.execute(
            "UPDATE media SET metadata=? WHERE material_id=?", (json.dumps(metadata), identifier)
        )
    old = [{"start": 0, "end": 1, "text": "old captions"}]
    uploaded = [{"start": 0, "end": 1, "text": "new uploaded captions"}]
    store.save_segments(space, identifier, old, "uploaded:old.vtt")
    job = jobs.enqueue("transcribe", space, settings, identifier)
    assert store.segments(space, identifier) == old

    def transcribe(*args):
        store.save_segments(space, identifier, uploaded, "uploaded:new.vtt")
        return {"segments": [{"start": 0, "end": 1, "text": "stale ASR"}], "alignment_notes": []}

    monkeypatch.setattr("learning_agent.media.transcribe_part", transcribe)
    with Worker(settings) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "cancelled"
    assert store.segments(space, identifier) == uploaded
    assert store.media(space, identifier)["transcript_state"] == "ready"
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM transcript_revisions").fetchone()[0] == 2


def test_transcript_result_saved_before_crash_is_reused(tmp_path, clip, monkeypatch):
    settings, store, space, jobs = setup(tmp_path, asr_provider="local")
    media = store.add_media(space, "clip.mkv", clip, ".mkv")
    identifier = media["material_id"]
    with store.connect() as db:
        metadata = json.loads(
            db.execute("SELECT metadata FROM media WHERE material_id=?", (identifier,)).fetchone()[
                0
            ]
        )
        metadata["has_audio"] = True
        db.execute(
            "UPDATE media SET metadata=? WHERE material_id=?", (json.dumps(metadata), identifier)
        )
    jobs.enqueue("transcribe", space, settings, identifier)
    segments = [{"start": 0, "end": 1, "text": "saved ASR"}]
    with Worker(settings):
        job = jobs.claim()
        jobs.save_part(job, "transcript", {"segments": segments, "alignment_notes": []})
    monkeypatch.setattr(
        "learning_agent.media.transcribe_part", lambda *args: pytest.fail("ASR must not repeat")
    )
    with Worker(settings) as worker:
        worker.drain()
    assert store.segments(space, identifier) == segments
    assert jobs.get(job["id"])["attempt"] == 2


def test_job_api_validation_offline_and_pause_cancel(tmp_path):
    settings, store, space, _jobs = setup(tmp_path, embedding_provider="ollama")
    client = TestClient(create_app(settings, Scripted([])))
    assert client.get(f"/api/spaces/{space}/jobs").json() == {"worker_online": False, "jobs": []}
    item = client.post(f"/api/spaces/{space}/retrieval").json()
    assert client.get(f"/api/jobs/{item['id']}").json()["state"] == "queued"
    assert client.post(f"/api/jobs/{item['id']}/pause").json()["state"] == "paused"
    assert client.post(f"/api/jobs/{item['id']}/cancel").json()["state"] == "cancelled"
    assert client.post(f"/api/jobs/{item['id']}/resume").status_code == 409
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/spaces/missing/jobs").status_code == 404
    assert client.get(f"/api/spaces/{store.space('other')['id']}/jobs").json()["jobs"] == []


def test_index_input_change_during_embedding_cannot_publish_stale_vectors(tmp_path):
    settings, store, space, jobs = setup(tmp_path, embedding_provider="ollama")
    material = store.add_material(space, "notes.txt", "attention")
    item = jobs.enqueue("index", space, settings)
    embedder = Batches(
        lambda _: store.set_source(space, material["id"], {"url": "https://example.com/"})
    )
    retriever = Retriever(store, settings, embedder)
    with Worker(settings, retriever) as worker:
        worker.drain()
    assert jobs.get(item["id"])["state"] == "cancelled"
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 0
    assert retriever.status(space)["state"] == "pending"


def test_failed_preview_never_serves_partial_output_and_can_retry(tmp_path, clip, monkeypatch):
    from learning_agent import media

    settings, _, space, jobs = setup(tmp_path)
    client = TestClient(create_app(settings, Scripted([])))
    item = client.post(f"/api/spaces/{space}/materials", files={"file": ("clip.mkv", clip)}).json()
    playback = f"/api/spaces/{space}/materials/{item['id']}/playback"
    real_prepare = media.prepare

    def fail(source, target, info):
        target.write_bytes(b"incomplete MP4")
        assert client.get(playback).status_code == 409
        raise media.MediaError("模拟中断")

    monkeypatch.setattr(media, "prepare", fail)
    with Worker(settings) as worker:
        worker.drain()
    assert jobs.get(item["job_id"])["state"] == "failed"
    assert client.get(playback).status_code == 409
    monkeypatch.setattr(media, "prepare", real_prepare)
    retried = client.post(f"/api/spaces/{space}/materials/{item['id']}/prepare").json()
    assert retried["id"] == item["job_id"]
    with Worker(settings) as worker:
        worker.drain()
    assert client.get(playback, headers={"Range": "bytes=0-127"}).status_code == 206


def test_sigkill_worker_does_not_allow_overlapping_child_writes(tmp_path):
    settings, _, _, _ = setup(tmp_path)
    marker = tmp_path / "child-ready"
    script = """
import sys
from pathlib import Path
from learning_agent.config import Settings
from learning_agent.media import command
from learning_agent.worker import Worker
child = "from pathlib import Path; import sys,time; Path(sys.argv[1]).touch(); time.sleep(30)"
with Worker(Settings(_env_file=None, data_root=Path(sys.argv[1]))):
    command([sys.executable, '-c', child, sys.argv[2]])
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(settings.data_root), str(marker)])
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.02)
        assert marker.exists()
        # Discover the child's group for test cleanup; it is separate from the killed worker.
        children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
        process.kill()
        process.wait(timeout=5)
        with (
            (settings.data_root / ".worker.lock").open("a+") as lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for child in locals().get("children", []):
            try:
                os.kill(int(child), 9)
            except ProcessLookupError:
                pass
