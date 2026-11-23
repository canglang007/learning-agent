import shutil

import pytest
from client_helpers import TestClient, close_for
from PIL import Image
from test_learning import Scripted, call

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.media import MediaError, command, parse_captions, probe
from learning_agent.store import Store
from learning_agent.tools import MaterialTools
from learning_agent.worker import Worker

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


@pytest.fixture
def clip(tmp_path):
    output = tmp_path / "clip.mp4"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:r=10:d=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            str(output),
        ]
    )
    return output.read_bytes()


def prepared(tmp_path, clip, model=None, **kwargs):
    config = Settings(_env_file=None, data_root=tmp_path / "db", **kwargs)
    model = model or Scripted([])
    close_for(tmp_path)
    client = TestClient(create_app(config, model))
    space = client.post("/api/spaces", json={"content": "media"}).json()["id"]
    upload = client.post(f"/api/spaces/{space}/materials", files={"file": ("colors.mp4", clip)})
    assert upload.status_code == 202
    material = upload.json()["id"]
    base = f"/api/spaces/{space}/materials/{material}"
    with Worker(config) as worker:
        worker.drain()
    assert client.get(base + "/media").json()["state"] == "ready"
    return client, space, material, base, config, model


def test_time_captions_frames_playback_range_and_scope(tmp_path, clip):
    # Explicitly unconfigured transcription: this test is about captions, not ASR.
    client, space, material, base, config, _ = prepared(tmp_path, clip, asr_provider="disabled")
    captions = b"WEBVTT\n\n00:00.000 --> 00:01.900\nred definition\n\n00:02.000 --> 00:04.000\nblue example\n"
    assert (
        client.post(base + "/captions", files={"file": ("colors.vtt", captions)}).status_code == 200
    )
    tools = MaterialTools(Store(config.data_root), space)
    seen = tools.execute(
        "inspect_media",
        {"material_id": material, "start_seconds": 0.0, "end_seconds": 4.0, "mode": "joint"},
    )
    assert len(seen["images"]) == 3
    assert seen["transcript_source"] == "uploaded:colors.vtt"
    assert seen["segments"][1]["start"] == 2
    one = tools.execute(
        "inspect_media",
        {
            "material_id": material,
            "start_seconds": 2.0,
            "end_seconds": 4.0,
            "mode": "joint",
            "frame_count": 1,
        },
    )
    assert len(one["images"]) == 1 and one["images"][0]["seconds"] == 3.0
    for image, channel in [(seen["images"][0], 0), (seen["images"][-1], 2)]:
        with Image.open(config.data_root / image["asset_path"]) as decoded:
            pixel = decoded.getpixel((50, 50))
            assert pixel[channel] > 240 and sum(pixel) < 270
    hit = tools.execute("search_materials", {"query": "blue"})["hits"][0]
    assert hit["start_seconds"] <= 2 and hit["end_seconds"] >= 4
    assert "blue example" in hit["excerpt"] and "#media=" in hit["source_url"]
    assert client.get(base + "/playback", headers={"Range": "bytes=0-99"}).status_code == 206
    other = client.post("/api/spaces", json={"content": "other"}).json()["id"]
    assert client.get(f"/api/spaces/{other}/materials/{material}/playback").status_code == 404
    assert (
        MaterialTools(Store(config.data_root), other).execute(
            "inspect_media",
            {"material_id": material, "start_seconds": 0.0, "end_seconds": 2.0, "mode": "joint"},
        )["error"]
        == "material_not_found"
    )
    assert client.post(base + "/transcribe").status_code == 503
    assert (
        client.post(
            base + "/captions",
            files={"file": ("bad.srt", b"1\n00:00:00,000 --> 00:01:00,000\ntoo long")},
        ).status_code
        == 422
    )


def test_media_agent_shared_budget_missing_captions_and_focus(tmp_path, clip):
    client, space, material, base, config, model = prepared(
        tmp_path, clip, max_images_per_turn=2, asr_provider="disabled"
    )
    conv = client.post(f"/api/spaces/{space}/conversations").json()["id"]
    args = {"material_id": material, "start_seconds": 0.0, "end_seconds": 4.0, "mode": "joint"}
    model.responses = [
        call("inspect_media", args),
        call("inspect_media", {**args, "end_seconds": 3.0}),
        {"role": "assistant", "content": "只看到颜色变化，没有讲话字幕。"},
    ]
    result = client.post(
        f"/api/conversations/{conv}/messages",
        json={
            "content": "解释",
            "focus": {"material_id": material, "start_seconds": 0, "end_seconds": 4},
        },
    ).json()
    assert result["state"] == "completed"
    events = client.get(f"/api/runs/{result['run_id']}").json()["events"]
    observations = [e["observation"] for e in events if e["type"] == "tool"]
    assert len(observations[0]["images"]) == 2 and observations[0]["segments"] == []
    assert observations[1]["frame_status"] == "budget_exhausted"
    assert len([m for m in model.inputs[-1] if m.get("attachments")]) == 2
    assert (
        client.post(
            f"/api/conversations/{conv}/messages",
            json={
                "content": "explain",
                "focus": {"material_id": material, "start_seconds": 2, "end_seconds": 1},
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/conversations/{conv}/messages",
            json={
                "content": "explain",
                "focus": {"material_id": material, "start_seconds": 0, "end_seconds": 10},
            },
        ).status_code
        == 422
    )
    assert (
        MaterialTools(Store(config.data_root), space, False).execute("inspect_media", args)[
            "images"
        ]
        == []
    )
    assert client.post(base + "/transcribe").status_code == 503


def test_recovery_and_no_audio_and_invalid_media(tmp_path, clip):
    client, space, material, base, config, _ = prepared(tmp_path, clip, asr_provider="local")
    assert client.post(base + "/transcribe").status_code == 422
    store = Store(config.data_root)
    # API startup cannot invalidate work owned by another process.
    with store.connect() as db:
        db.execute("UPDATE media SET state='processing' WHERE material_id=?", (material,))
        db.execute("DELETE FROM jobs WHERE material_id=?", (material,))
    create_app(config, Scripted([]))
    assert store.media(space, material)["state"] == "processing"
    with Worker(config) as worker:
        worker.drain()
    assert store.media(space, material)["state"] == "ready"
    assert client.post(base + "/prepare").status_code == 202
    with Worker(config) as worker:
        worker.drain()
    assert store.media(space, material)["state"] == "ready"
    assert (
        client.post(
            f"/api/spaces/{space}/materials", files={"file": ("bad.mp4", b"not video")}
        ).status_code
        == 422
    )


def test_srt_vtt_timestamps_and_no_fabrication():
    segments = parse_captions(b"1\n01:02:03,125 --> 01:02:04,500\nfirst\nsecond\n", 4000)
    assert segments == [{"start": 3723.125, "end": 3724.5, "text": "first\nsecond"}]
    with pytest.raises(MediaError):
        parse_captions(b"plain text without time", 30)
    with pytest.raises(MediaError):
        parse_captions(b"00:00.000 --> 00:60.000\nbad", 100)


@pytest.mark.parametrize("duration", [61, 60.5])
def test_transcription_chunk_offsets(tmp_path, monkeypatch, duration):
    import httpx
    from pydantic import SecretStr

    from learning_agent import media

    offsets = []

    def fake_command(args, timeout=180):
        # Decoder-independent protocol test; real ASR validated separately.
        from pathlib import Path

        Path(args[-1]).write_bytes(b"pcm fixture")
        if "-ss" in args:
            offsets.append(float(args[args.index("-ss") + 1]))
        return b""

    monkeypatch.setattr(media, "command", fake_command)

    def response(request):
        assert (
            b"verbose_json" in request.content and b"timestamp_granularities[]" in request.content
        )
        return httpx.Response(200, json={"segments": [{"start": 0.2, "end": 0.8, "text": "word"}]})

    real_client = httpx.Client
    monkeypatch.setattr(
        media.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(response))
    )
    settings = Settings(
        _env_file=None,
        asr_provider="compatible",
        asr_model="whisper-1",
        asr_api_key=SecretStr("dummy"),
        asr_context_seconds=0,
    )
    result = media.transcribe(tmp_path / "audio.wav", duration, settings)
    assert (
        offsets == [0, 60]
        and result["segments"][1]["start"] == 60.2
        and result["segments"][1]["end"] == min(60.8, duration)
    )
    if duration == 60.5:
        assert result["alignment_notes"][0]["reported_end"] == 60.8
        assert result["alignment_notes"][0]["playable_end"] == duration


def test_local_asr_accepts_decoded_waveform_and_failure_preserves_captions(tmp_path, monkeypatch):
    import wave
    from types import SimpleNamespace

    from learning_agent import media

    whisper = pytest.importorskip("faster_whisper")
    np = pytest.importorskip("numpy")
    source = tmp_path / "tone.wav"
    with wave.open(str(source), "wb") as pcm:
        pcm.setnchannels(1)
        pcm.setsampwidth(2)
        pcm.setframerate(16000)
        pcm.writeframes(b"\0\0" * 16000)

    class WaveformModel:
        def transcribe(self, audio, **kwargs):
            # Regression: passing filenames invokes FW1.2.1's broken PyAV19 decoder.
            assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
            assert len(audio) == 16000
            return iter([SimpleNamespace(start=0, end=1.4, text="speech")]), None

    monkeypatch.setattr(whisper, "WhisperModel", lambda *args, **kwargs: WaveformModel())
    monkeypatch.setattr(media, "_models", {})
    config = Settings(_env_file=None, data_root=tmp_path / "db", asr_provider="local")
    result = media.transcribe(source, 1.0, config)
    assert result["segments"] == [{"start": 0.0, "end": 1.0, "text": "speech"}]
    assert result["alignment_notes"][0]["reported_end"] == 1.4
    close_for(tmp_path)
    client = TestClient(create_app(config, Scripted([])))
    space = client.post("/api/spaces", json={"content": "audio"}).json()["id"]
    material = client.post(
        f"/api/spaces/{space}/materials", files={"file": ("tone.wav", source.read_bytes())}
    ).json()["id"]
    base = f"/api/spaces/{space}/materials/{material}"
    assert client.get(base + "/media").json()["format"] == "audio"
    assert client.post(base + "/transcribe").status_code == 202
    with Worker(config) as worker:
        worker.drain()
    assert client.get(base + "/segments").json()[0]["text"] == "speech"

    def fail(*args):
        raise MediaError("识别失败，请重试")

    monkeypatch.setattr(media, "transcribe_part", fail)
    assert client.post(base + "/transcribe").status_code == 202
    with Worker(config) as worker:
        worker.drain()
    assert client.get(base + "/media").json()["transcript_state"] == "failed"
    assert client.get(base + "/segments").json()[0]["text"] == "speech"


def test_playback_plan_grades_container_and_codec():
    """能不能直接播看"容器 + 编码"，不看扩展名猜。"""
    from learning_agent import media

    v_h264 = {"codec_name": "h264", "pix_fmt": "yuv420p"}
    v_av1 = {"codec_name": "av1", "pix_fmt": "yuv420p"}
    v_vp9 = {"codec_name": "vp9", "pix_fmt": "yuv420p"}
    v_prores = {"codec_name": "prores", "pix_fmt": "yuv422p10le"}
    a_aac = {"codec_name": "aac"}
    a_opus = {"codec_name": "opus"}

    assert media.playback_plan(".mp4", v_h264, a_aac) == "direct"
    # AV1 装进 mp4：以前被"必须 h264"挡掉，白跑一次全量重编码。
    assert media.playback_plan(".mp4", v_av1, a_aac) == "direct"
    assert media.playback_plan(".webm", v_vp9, a_opus) == "direct"
    # mkv 装 h264：只差一层壳，换容器即可，不该重编码。
    assert media.playback_plan(".mkv", v_h264, a_aac) == "remux"
    # 编码本身浏览器不认：这类才需要重编码。
    assert media.playback_plan(".mp4", v_prores, a_aac) == "transcode"
    assert media.playback_plan(".mp4", {"codec_name": "h264", "pix_fmt": "yuv444p"}, a_aac) == (
        "transcode"
    )


def test_h264_in_mkv_is_remuxed_not_reencoded(tmp_path):
    """mkv 装 h264 的文件要能直接进产品并播放，走 remux 而不是重编码。"""
    source = tmp_path / "colors.mkv"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x120:r=10:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            str(source),
        ]
    )
    config = Settings(_env_file=None, data_root=tmp_path / "db", asr_provider="disabled")
    close_for(tmp_path)
    client = TestClient(create_app(config, Scripted([])))
    space = client.post("/api/spaces", json={"content": "mkv"}).json()["id"]
    upload = client.post(
        f"/api/spaces/{space}/materials", files={"file": ("colors.mkv", source.read_bytes())}
    )
    assert upload.status_code == 202
    material = upload.json()["id"]
    base = f"/api/spaces/{space}/materials/{material}"
    with Worker(config) as worker:
        worker.drain()

    store = Store(config.data_root)
    info = store.media(space, material)
    assert info["state"] == "ready"
    assert info["playback"] == "remux"
    assert info["browser_playable"] is False
    preview = info["preview_path"]
    assert preview.endswith(".mp4") and preview != info["original_path"]
    # 转出来的预览本身必须是浏览器可直接播的，否则等于白转。
    assert probe(config.data_root / preview)["browser_playable"] is True
    assert client.get(base + "/playback", headers={"Range": "bytes=0-99"}).status_code == 206


def test_oversized_video_is_downscaled_not_rejected():
    """超出 4K 不该让用户先去转码：标成 transcode，由 prepare 缩到 1280 宽。"""
    from learning_agent import media

    v_h264 = {"codec_name": "h264", "pix_fmt": "yuv420p"}
    assert media.playback_plan(".mp4", v_h264, {"codec_name": "aac"}, oversized=True) == "transcode"
    assert media.playback_plan(".mp4", v_h264, {"codec_name": "aac"}) == "direct"


def test_content_type_follows_real_container():
    """回放不能一律报 video/mp4：直接可播时给的就是原文件。"""
    from learning_agent import media

    assert media.content_type("materials/x/original.webm") == "video/webm"
    assert media.content_type("materials/x/original.mkv") == "video/x-matroska"
    assert media.content_type("materials/x/original.m4a") == "audio/mp4"
    assert media.content_type("materials/x/original.mp4") == "video/mp4"


def test_avi_and_webm_are_accepted_instead_of_told_to_convert(tmp_path):
    """mp4 之外的真实文件（avi / webm）应当能直接入库并播放。

    以前它们被后缀白名单挡在门外、被要求"先转成 mp4"——把工具的麻烦推给用户。
    avi 里的 mpeg4 编码浏览器不认，走重编码；webm 的 vp9+opus 本来就能播，原样提供。
    """
    avi = tmp_path / "colors.avi"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:r=10:d=2",
            "-c:v",
            "mpeg4",
            "-f",
            "avi",
            str(avi),
        ]
    )
    webm = tmp_path / "colors.webm"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=teal:s=160x120:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "200k",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            str(webm),
        ]
    )
    config = Settings(_env_file=None, data_root=tmp_path / "db", asr_provider="disabled")
    close_for(tmp_path)
    client = TestClient(create_app(config, Scripted([])))
    space = client.post("/api/spaces", json={"content": "formats"}).json()["id"]

    ids = {}
    for name, path in (("colors.avi", avi), ("colors.webm", webm)):
        upload = client.post(
            f"/api/spaces/{space}/materials", files={"file": (name, path.read_bytes())}
        )
        assert upload.status_code == 202, upload.text
        ids[name] = upload.json()["id"]
    with Worker(config) as worker:
        worker.drain()

    store = Store(config.data_root)
    avi_info = store.media(space, ids["colors.avi"])
    assert avi_info["state"] == "ready"
    assert avi_info["playback"] == "transcode"
    preview = avi_info["preview_path"]
    assert preview.endswith(".mp4") and probe(config.data_root / preview)["browser_playable"]

    webm_info = store.media(space, ids["colors.webm"])
    assert webm_info["state"] == "ready"
    # vp9 + opus 浏览器本来就能播：不该白转一遍，原文件直接给。
    assert webm_info["playback"] == "direct"
    assert webm_info["preview_path"] == webm_info["original_path"]
    response = client.get(
        f"/api/spaces/{space}/materials/{ids['colors.webm']}/playback",
        headers={"Range": "bytes=0-99"},
    )
    assert response.status_code == 206
    assert response.headers["content-type"] == "video/webm"


def test_ass_and_utf16_subtitles_are_converted(tmp_path):
    """B 站的 .ass、UTF-16 的 .srt 都要能用，不能让用户自己先转成 UTF-8 VTT。"""
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:01,800\n你好 world\n", encoding="utf-8")
    ass = tmp_path / "out.ass"
    command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(srt), str(ass)])
    segments = parse_captions(ass.read_bytes(), 30, ".ass")
    assert [s["text"] for s in segments] == ["你好 world"]
    assert 0.3 < segments[0]["start"] < 0.8 and 1.5 < segments[0]["end"] < 2.2

    utf16 = ("1\n00:00:02,000 --> 00:00:03,000\n中文十六位\n").encode("utf-16")
    converted = parse_captions(utf16, 30, ".srt")
    assert [s["text"] for s in converted] == ["中文十六位"]
