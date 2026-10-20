"""Time-aligned media observations. ffmpeg/ffprobe are system dependencies."""

import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import wave
from contextvars import ContextVar
from pathlib import Path

import httpx

MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".m4a", ".wav", ".ogg", ".flac"}
_asr_lock = threading.Lock()
_models = {}
_worker_lock_fd = ContextVar("worker_lock_fd", default=None)
_render_process_group = ContextVar("render_process_group", default=False)


class MediaError(ValueError):
    pass


def command(args, timeout=180):
    if not shutil.which(args[0]):
        raise MediaError("需要在本机安装 ffmpeg（包含 ffprobe）后处理音视频")
    try:
        lock_fd = _worker_lock_fd.get()
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=not _render_process_group.get(),
            pass_fds=() if lock_fd is None else (lock_fd,),
        )
        try:
            stdout, _ = process.communicate(timeout=timeout)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, args)
            return stdout
        finally:
            if process.poll() is None:
                try:
                    if _render_process_group.get():
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    except (subprocess.SubprocessError, OSError):
        raise MediaError("媒体解码失败或超时，请确认文件完整且编码受支持") from None


def probe(path, max_seconds=14400):
    try:
        data = json.loads(
            command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )
        duration = float(data.get("format", {}).get("duration", 0))
    except (ValueError, TypeError, KeyError):
        raise MediaError("无法读取有效媒体时长，请检查文件是否完整") from None
    if not math.isfinite(duration) or not 0.1 <= duration <= max_seconds:
        raise MediaError(f"媒体时长须为0.1秒到{max_seconds / 3600:g}小时")
    video = next(
        (
            s
            for s in data["streams"]
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if not video and not audio:
        raise MediaError("文件没有可用的音视频轨道")
    if video and int(video.get("width", 0)) * int(video.get("height", 0)) > 3840 * 2160:
        raise MediaError("初版支持最高4K视频，请先降低分辨率")
    return {
        "duration": duration,
        "has_video": bool(video),
        "has_audio": bool(audio),
        "browser_playable": Path(path).suffix.lower() in {".mp4", ".m4a"}
        and "mp4" in data.get("format", {}).get("format_name", "")
        and (not video or (video.get("codec_name") == "h264" and video.get("pix_fmt") == "yuv420p"))
        and (not audio or audio.get("codec_name") == "aac"),
    }


def prepare(source, target, info):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source)]
    if info["has_video"]:
        args += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
        ]
    else:
        args += ["-vn", "-c:a", "aac", "-movflags", "+faststart"]
    command([*args, str(target)], timeout=max(600, info["duration"] * 2))


def frame(source, target, seconds):
    # Readers must not observe a half-written cached PNG from another conversation.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".png", delete=False) as file:
        temporary = Path(file.name)
    try:
        _extract_frame(source, temporary, seconds)
        from learning_agent.assets import png_size

        if png_size(temporary) is None:
            raise MediaError("抽帧结果不可解码，未发布缓存")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_frame(source, target, seconds):
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            str(seconds),
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(1280,iw)':-2",
            str(target),
        ]
    )
    if not target.exists() or target.stat().st_size == 0:
        raise MediaError("该时间点没有可解码画面，请选择更早的时间")


def valid_segments(segments, duration, allow_empty=False):
    result = []
    for segment in segments:
        try:
            start, end = float(segment["start"]), float(segment["end"])
            text = str(segment["text"]).strip()
        except (ValueError, TypeError, KeyError):
            raise MediaError("字幕片段缺少有效的起止时间或文字") from None
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start < min(end, duration)
            and end <= duration + 0.25
        ):
            raise MediaError("字幕时间必须在媒体时长范围内，且结束时间大于开始时间")
        if text:
            result.append({"start": start, "end": min(end, duration), "text": text[:4000]})
    if not result and not allow_empty:
        raise MediaError("没有识别到带时间的字幕；不会用整段文字伪造时间轴")
    if len(result) > 20000:
        raise MediaError("字幕片段数量超过初版限制")
    return sorted(result, key=lambda s: (s["start"], s["end"]))


def parse_captions(raw, duration):
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        raise MediaError("字幕需要 UTF-8 编码的 SRT 或 WebVTT") from None
    # SRT and WebVTT timestamps, including WebVTT's optional hour component/settings.
    stamp = r"((?:\d+:)?\d{2}:\d{2}[.,]\d{3})"
    pattern = re.compile(
        r"^" + stamp + r"\s+-->\s+" + stamp + r"[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    def seconds(value):
        parts = value.replace(",", ".").split(":")
        if float(parts[-1]) >= 60 or (len(parts) == 3 and int(parts[-2]) >= 60):
            raise MediaError("字幕时间格式错误")
        return sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))

    segments = [
        {"start": seconds(m[1]), "end": seconds(m[2]), "text": re.sub(r"<[^>]*>", "", m[3]).strip()}
        for m in pattern.finditer(text)
    ]
    return valid_segments(segments, duration)


def asr_parts(duration, settings):
    seconds = (
        settings.asr_local_part_seconds
        if settings.asr_provider == "local"
        else settings.asr_api_part_seconds
    )
    context = settings.asr_context_seconds
    return [
        {
            "index": i,
            "start": start,
            "end": min(start + seconds, duration),
            "decode_start": max(0, start - context),
            "decode_end": min(start + seconds + context, duration),
        }
        for i, start in enumerate(range(0, math.ceil(duration), seconds))
    ]


def transcribe_part(path, part, settings, directory, before_remote=None):
    """Decode only this bounded window. Return absolute source timestamps, including raw estimates."""
    if settings.asr_provider == "disabled":
        raise MediaError("尚未配置语音识别，可先上传SRT/VTT字幕")
    if settings.asr_provider == "compatible" and not settings.asr_api_key.get_secret_value():
        raise MediaError("兼容转写需要单独配置 LA_ASR_API_KEY，不能假定聊天密钥支持ASR")
    directory.mkdir(parents=True, exist_ok=True)
    audio = directory / f"part-{part['index']}.wav"
    window = part["decode_end"] - part["decode_start"]
    try:
        command(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                str(part["decode_start"]),
                "-i",
                str(path),
                "-t",
                str(window),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio),
            ],
            timeout=max(180, window * 2),
        )
        if settings.asr_provider == "local":
            with _asr_lock:
                try:
                    import numpy as np
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise MediaError("请先运行 uv sync --extra asr 安装本地转写依赖") from None
                try:
                    if settings.asr_model not in _models:
                        _models[settings.asr_model] = WhisperModel(
                            settings.asr_model, device="cpu", compute_type="int8"
                        )
                    with wave.open(str(audio), "rb") as pcm:
                        if pcm.getnframes() > math.ceil(window * 16000) + 16000:
                            raise MediaError("解码音频超出当前分片范围")
                        samples = (
                            np.frombuffer(pcm.readframes(pcm.getnframes()), dtype="<i2").astype(
                                np.float32
                            )
                            / 32768.0
                        )
                    segments, _ = _models[settings.asr_model].transcribe(samples, vad_filter=True)
                    result = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
                except MediaError:
                    raise
                except Exception:  # noqa: BLE001 -- native/model-download boundary; redact details
                    raise MediaError(
                        "本地转写失败，请检查模型下载和CPU内存；可上传字幕继续"
                    ) from None
        else:
            try:
                with httpx.Client(timeout=180) as client, audio.open("rb") as file:
                    if before_remote:
                        before_remote()
                    response = client.post(
                        settings.asr_base_url.rstrip("/") + "/audio/transcriptions",
                        headers={
                            "Authorization": "Bearer " + settings.asr_api_key.get_secret_value()
                        },
                        data={
                            "model": settings.asr_model,
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": "segment",
                        },
                        files={"file": ("chunk.wav", file, "audio/wav")},
                    )
                    response.raise_for_status()
                    body = response.json()
                    result = body["segments"]
                    if not isinstance(result, list) or (
                        not result and str(body.get("text", "")).strip()
                    ):
                        raise ValueError("missing timed segments")
            except (httpx.HTTPError, ValueError, TypeError, KeyError):
                raise MediaError("转写API请求失败或未返回有效时间片段，请检查服务兼容性") from None
        aligned = []
        for segment in result:
            try:
                start = float(segment["start"]) + part["decode_start"]
                end = float(segment["end"]) + part["decode_start"]
                text = str(segment["text"]).strip()
            except (ValueError, KeyError, TypeError):
                raise MediaError("语音识别返回了无效时间片段") from None
            if not (math.isfinite(start) and math.isfinite(end) and start < end):
                raise MediaError("语音识别返回了无效时间戳")
            aligned.append({"start": start, "end": end, "text": text})
        return {"part": part, "segments": aligned}
    finally:
        audio.unlink(missing_ok=True)


def merge_transcripts(results, duration):
    segments, notes = [], []
    for result in results:
        part = result["part"]
        for segment in result["segments"]:
            start, end = segment["start"], segment["end"]
            a, b = max(part["decode_start"], start), min(part["decode_end"], end, duration)
            if a != start or b != end or a >= b:
                notes.append(
                    {
                        "kind": "window_clipping",
                        "part": part["index"],
                        "reported_start": start,
                        "reported_end": end,
                        "playable_start": a if a < b else None,
                        "playable_end": b if a < b else None,
                    }
                )
            if a >= b:
                continue
            midpoint = (a + b) / 2
            if not part["start"] <= midpoint < part["end"]:
                continue
            if a < part["start"] or b > part["end"]:
                notes.append(
                    {"kind": "crosses_part_boundary", "part": part["index"], "start": a, "end": b}
                )
            segments.append(
                {"start": a, "end": b, "text": segment["text"], "parts": {part["index"]}}
            )
    merged = []

    def contained(short, long):
        # Exact tokens only: no similarity score or guessed semantic correction.
        def tokens(text):
            return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]|[^\w\s]", text.casefold())

        a, b = tokens(short["text"]), tokens(long["text"])
        return (
            len(a) >= 3
            and short["start"] >= long["start"] - 0.25
            and short["end"] <= long["end"] + 0.25
            and any(b[i : i + len(a)] == a for i in range(len(b) - len(a) + 1))
        )

    for segment in sorted(segments, key=lambda s: (s["start"], s["end"])):
        neighbors = [
            s
            for s in reversed(merged[-10:])
            if s["parts"].isdisjoint(segment["parts"]) and s["end"] > segment["start"]
        ]
        duplicate = next(
            (s for s in neighbors if s["text"] == segment["text"]),
            None,
        )
        if duplicate:
            duplicate["end"] = max(duplicate["end"], segment["end"])
            duplicate["parts"].update(segment["parts"])
            notes.append(
                {
                    "kind": "boundary_duplicate_removed",
                    "start": segment["start"],
                    "end": segment["end"],
                }
            )
        elif any(contained(segment, s) for s in neighbors):
            notes.append(
                {
                    "kind": "boundary_duplicate_removed",
                    "start": segment["start"],
                    "end": segment["end"],
                }
            )
        else:
            for earlier in neighbors:
                if contained(earlier, segment):
                    merged.remove(earlier)
                    notes.append(
                        {
                            "kind": "boundary_duplicate_removed",
                            "start": earlier["start"],
                            "end": earlier["end"],
                        }
                    )
            merged.append(segment)
    return {
        "segments": valid_segments(merged, duration, allow_empty=True),
        "alignment_notes": notes,
    }


def transcribe(path, duration, settings):
    """Bounded standalone driver. Durable callers save every transcribe_part result themselves."""
    with tempfile.TemporaryDirectory(prefix="learning-asr-") as directory:
        return merge_transcripts(
            [
                transcribe_part(path, part, settings, Path(directory))
                for part in asr_parts(duration, settings)
            ],
            duration,
        )
