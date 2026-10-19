import wave
from types import SimpleNamespace

import httpx
import pytest

from learning_agent import media
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.store import Store
from learning_agent.worker import Worker


def audio(tmp_path, seconds):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0\0" * int(seconds * 16000))
    return path


def test_local_decoder_only_reads_one_window_and_preserves_global_time(tmp_path, monkeypatch):
    whisper = pytest.importorskip("faster_whisper")
    lengths = []

    class Model:
        def transcribe(self, samples, **kwargs):
            lengths.append(len(samples))
            middle = len(samples) / 32000
            return iter(
                [SimpleNamespace(start=middle - 0.1, end=middle + 0.1, text="speech")]
            ), None

    monkeypatch.setattr(whisper, "WhisperModel", lambda *a, **k: Model())
    monkeypatch.setattr(media, "_models", {})
    settings = Settings(
        _env_file=None, asr_provider="local", asr_local_part_seconds=10, asr_context_seconds=2
    )
    result = media.transcribe(audio(tmp_path, 25), 25, settings)
    assert lengths == [12 * 16000, 14 * 16000, 7 * 16000]
    assert len(result["segments"]) == 3
    assert result["segments"][-1]["start"] > 20
    assert not list(tmp_path.glob("part-*.wav"))


def setup(tmp_path, provider="local"):
    path = audio(tmp_path, 25)
    settings = Settings(
        _env_file=None,
        data_root=tmp_path / "db",
        asr_provider=provider,
        asr_local_part_seconds=10,
        asr_api_part_seconds=10,
        asr_context_seconds=2,
    )
    store = Store(settings.data_root)
    space = store.space("course")["id"]
    item = store.add_media_file(space, "audio.wav", path, ".wav", move=False)
    jobs = Jobs(store)
    job = jobs.enqueue("transcribe", space, settings, item["material_id"])
    return settings, store, space, item, jobs, job


def test_pause_and_restart_only_process_missing_parts(tmp_path, monkeypatch):
    settings, store, space, item, jobs, job = setup(tmp_path)
    calls = []

    def recognize(path, part, settings, directory, before_remote=None):
        calls.append(part["index"])
        if len(calls) == 1:
            jobs.action(job["id"], "pause")
        return {
            "part": part,
            "segments": [
                {"start": part["start"] + 0.1, "end": part["end"] - 0.1, "text": "speech"}
            ],
        }

    monkeypatch.setattr(media, "transcribe_part", recognize)
    with Worker(settings) as worker:
        worker.drain()
    assert calls == [0]
    assert jobs.get(job["id"])["state"] == "paused"
    assert store.segments(space, item["material_id"]) == []  # Candidate is not a full transcript.
    jobs.action(job["id"], "resume")
    with Worker(settings) as worker:
        worker.drain()
    assert calls == [0, 1, 2]
    assert jobs.get(job["id"])["processed_seconds"] == 25
    assert jobs.get(job["id"])["state"] == "succeeded"
    assert jobs.part(job, "asr:0")["created_attempt"] == 1
    assert jobs.part(job, "asr:1")["created_attempt"] == 2


def test_unknown_remote_part_requires_retry_but_saved_parts_are_reused(tmp_path, monkeypatch):
    settings, _, _, _, jobs, job = setup(tmp_path, "compatible")
    calls = []

    def recognize(path, part, settings, directory, before_remote=None):
        calls.append(part["index"])
        before_remote()
        if calls == [0, 1]:
            raise media.MediaError("connection lost")
        return {
            "part": part,
            "segments": [{"start": part["start"], "end": part["end"], "text": "saved"}],
        }

    monkeypatch.setattr(media, "transcribe_part", recognize)
    with Worker(settings) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "failed"
    assert "remote_result_unknown" in jobs.get(job["id"])["error"]
    with Worker(settings) as worker:
        worker.drain()
    assert calls == [0, 1]
    jobs.action(job["id"], "retry")
    with Worker(settings) as worker:
        worker.drain()
    assert calls == [0, 1, 1, 2]
    assert jobs.get(job["id"])["state"] == "succeeded"


def test_silent_parts_finish_without_perpetual_retries(tmp_path, monkeypatch):
    settings, store, space, item, jobs, job = setup(tmp_path)
    monkeypatch.setattr(
        media, "transcribe_part", lambda path, part, *args: {"part": part, "segments": []}
    )
    with Worker(settings) as worker:
        worker.drain()
    assert jobs.get(job["id"])["state"] == "succeeded"
    assert store.media(space, item["material_id"])["transcript_state"] == "silent"
    assert store.segments(space, item["material_id"]) == []


def test_context_time_offsets_past_one_hour_and_missing_segments_are_not_silence(
    tmp_path, monkeypatch
):
    settings = Settings(_env_file=None, asr_provider="compatible", asr_api_key="dummy")
    part = media.asr_parts(3700, settings)[60]
    assert part["start"] == 3600 and part["decode_start"] == 3595
    monkeypatch.setattr(
        media, "command", lambda args, **kwargs: (tmp_path / "part-60.wav").write_bytes(b"fixture")
    )
    body = {"segments": [{"start": 5.2, "end": 6.2, "text": "after one hour"}]}
    real_client = httpx.Client
    monkeypatch.setattr(
        media.httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ),
    )
    result = media.transcribe_part(tmp_path / "source", part, settings, tmp_path)
    assert media.merge_transcripts([result], 3700)["segments"][0]["start"] == 3600.2
    body.clear()
    body["text"] = "text without timestamps"
    with pytest.raises(media.MediaError, match="有效时间片段"):
        media.transcribe_part(tmp_path / "source", part, settings, tmp_path)


def test_boundary_ownership_and_exact_overlap_deduplication():
    results = [
        {
            "part": {"index": 0, "start": 0, "end": 10, "decode_start": 0, "decode_end": 12},
            "segments": [
                {"start": 9, "end": 11, "text": "boundary"},
                {"start": 8, "end": 10.5, "text": "same"},
            ],
        },
        {
            "part": {"index": 1, "start": 10, "end": 20, "decode_start": 8, "decode_end": 20},
            "segments": [
                {"start": 9, "end": 11, "text": "boundary"},
                {"start": 9, "end": 12, "text": "same"},
            ],
        },
    ]
    result = media.merge_transcripts(results, 20)
    assert [s["text"] for s in result["segments"]] == ["same", "boundary"]
    assert result["segments"][0]["end"] == 12
    assert any(n["kind"] == "crosses_part_boundary" for n in result["alignment_notes"])


def test_real_failure_shape_contained_boundary_tail_is_removed_but_repetitions_survive():
    results = [
        {
            "part": {"index": 0, "start": 0, "end": 300, "decode_start": 0, "decode_end": 305},
            "segments": [
                {"start": 298.2, "end": 300.8, "text": "rotations keep inner products unchanged."}
            ],
        },
        {
            "part": {"index": 1, "start": 300, "end": 600, "decode_start": 295, "decode_end": 605},
            "segments": [
                {"start": 299.56, "end": 300.8, "text": "inner products unchanged."},
                {"start": 300, "end": 300.7, "text": "inner products change."},
                {"start": 301, "end": 302, "text": "inner products unchanged."},
            ],
        },
    ]
    result = media.merge_transcripts(results, 600)
    assert [s["text"] for s in result["segments"]] == [
        "rotations keep inner products unchanged.",
        "inner products change.",
        "inner products unchanged.",
    ]
    assert any(n["kind"] == "boundary_duplicate_removed" for n in result["alignment_notes"])
