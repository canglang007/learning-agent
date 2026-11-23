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

# 这份清单只用来判断"这是不是一份音视频"，不是准入门槛。真正的准入由 ffprobe
# 决定（能探出轨道就接收）。以前它被当成白名单用，于是 .avi / .ts / .wmv 的用户
# 被告知"请先转成 mp4"——把工具的麻烦推给用户，而 ffmpeg 本来就能读这些格式。
# 现在清单覆盖 ffmpeg 常见的容器与编码，剩下判断交给探测。
MEDIA_SUFFIXES = {
    # 视频容器
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".f4v",
    ".mpg", ".mpeg", ".mts", ".m2ts", ".ts", ".3gp", ".3g2", ".ogv", ".rmvb",
    ".vob", ".asf", ".divx", ".dv",
    # 音频容器与编码
    ".mp3", ".m4a", ".m4b", ".aac", ".wav", ".wave", ".flac", ".ogg", ".oga",
    ".opus", ".wma", ".aiff", ".aif", ".ac3", ".amr", ".ape", ".wv", ".mka",
}  # fmt: skip
# 字幕同理：官方渠道给的不一定是 SRT/VTT——B 站给 .ass、YouTube 的 ttml/srv3、
# 有些课程给 .sbv / MicroDVD .sub。让用户自己转格式属于同一类"迁就"。
SUBTITLE_SUFFIXES = {".srt", ".vtt", ".ass", ".ssa", ".ttml", ".dfxp", ".sbv", ".sub", ".smi"}
# 这两种本身就是带时间的纯文本，直接解析（保留原始时间戳，不经 ffmpeg 重排）。
_TEXT_TIMED_SUBTITLES = {".srt", ".vtt"}
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
}
# 浏览器能不能直接播，取决于"容器 + 编码"，不是扩展名。同一个 mkv 里装 h264+aac，
# 差的只是一层壳（remux 秒级）；一个 mp4 里装 ProRes，反而播不了（必须重编码）。
# 早先的判据要求"扩展名 mp4 且 h264 且 yuv420p 且 aac"，于是 720p 的 AV1 mp4、
# 或 h264 装进 mkv，全被判成不可播，白跑一次 libx264 全量重编码——长课程在 1 核上
# 要几十分钟。这里按能力分级：direct 直接播、remux 只换壳、transcode 才重编码。
DIRECT_CONTAINERS = {"mp4", "m4a", "m4v", "mov", "webm"}
DIRECT_VIDEO_CODECS = {"h264", "vp8", "vp9", "av1"}
DIRECT_AUDIO_CODECS = {"aac", "mp3", "opus", "vorbis"}
_MP4_SAFE_AUDIO = {"aac", "mp3"}
_asr_lock = threading.Lock()
_models = {}
_worker_lock_fd = ContextVar("worker_lock_fd", default=None)
_render_process_group = ContextVar("render_process_group", default=False)


class MediaError(ValueError):
    pass


def asr_available():
    """True when the optional local transcription engine can be imported."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:  # noqa: BLE001 -- optional dependency boundary
        return False
    return True


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


def playback_plan(suffix, video, audio, oversized=False):
    """能否在浏览器里直接播：direct 直接播 / remux 只换壳 / transcode 重编码。

    ``suffix`` 用保存下来的文件扩展名判断容器——ffprobe 会把 mkv 报成
    ``matroska,webm``，只看 format_name 会把 mkv 误判成 webm。
    ``oversized`` 指超出 4K：这类要缩尺寸，只能重编码，换壳解决不了。
    """
    video_codec = (video or {}).get("codec_name")
    audio_codec = (audio or {}).get("codec_name")
    if video and video_codec not in DIRECT_VIDEO_CODECS:
        return "transcode"
    if video and video_codec == "h264" and video.get("pix_fmt") not in {"yuv420p", "yuvj420p"}:
        return "transcode"
    if audio and audio_codec not in DIRECT_AUDIO_CODECS:
        return "transcode"
    if oversized:
        return "transcode"
    container = str(suffix).lower().strip()
    # 传进来的可能是 ".mkv" 也可能带文件名（"colors.mkv"）：Path(".mkv").suffix 是空，
    # 点开头的字符串会被当成隐藏文件名，所以直接按最后一个点切。
    container = container[container.rfind(".") + 1 :]
    return "direct" if container in DIRECT_CONTAINERS else "remux"


def content_type(path):
    """按实际文件给 Content-Type，不要假定一切都是 mp4。"""
    return CONTENT_TYPES.get(Path(str(path)).suffix.lower(), "application/octet-stream")


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
    area = int(video.get("width", 0)) * int(video.get("height", 0)) if video else 0
    # 超出 4K 不拒绝：prepare 本来就会缩到 1280 宽，要求用户先自己转码属于把工具的
    # 麻烦推回去。这里只把它降级成 transcode。
    playback = playback_plan(Path(path).suffix, video, audio, oversized=area > 3840 * 2160)
    return {
        "duration": duration,
        "has_video": bool(video),
        "has_audio": bool(audio),
        "playback": playback,
        "browser_playable": playback == "direct",
        "video_codec": (video or {}).get("codec_name"),
        "audio_codec": (audio or {}).get("codec_name"),
        "width": int(video.get("width", 0)) if video else 0,
        "height": int(video.get("height", 0)) if video else 0,
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


def remux(source, target, info):
    """只换容器：视频流原样复制，音频在 mp4 里不通用时转 aac。

    比 ``prepare`` 快两个数量级（秒级 vs 分钟级），也不损失画质——mkv 装 h264
    这种"只差一层壳"的文件没必要重编码。
    """
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source)]
    if info["has_video"]:
        args += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy"]
    else:
        args += ["-vn"]
    if info["has_audio"]:
        if (info.get("audio_codec") or "") in _MP4_SAFE_AUDIO:
            args += ["-c:a", "copy"]
        else:
            args += ["-c:a", "aac", "-b:a", "128k"]
    args += ["-movflags", "+faststart"]
    command([*args, str(target)], timeout=max(300, info["duration"]))


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


def sample_times(start, end, count, *, duration=None):
    """Return stable interior keyframe times for a bounded media window.

    Interior points avoid duplicate boundary frames and keep the visual evidence
    evenly distributed when a joint inspection has a small image budget.
    """
    if count <= 0 or end <= start:
        return []
    limit = end if duration is None else min(end, duration)
    span = limit - start
    if span <= 0:
        return []
    return [
        round(min(limit - 0.001, start + span * (index + 0.5) / count), 3) for index in range(count)
    ]


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


def _subtitle_as_webvtt(raw, suffix):
    """把非 SRT/VTT 的字幕（或非 UTF-8 编码的字幕）先交给 ffmpeg 转成 VTT。

    官方渠道给的不一定是 SRT/VTT：B 站给 .ass、YouTube 给 .ttml/.srv3、有的课程给
    .sbv，也有 GBK/UTF-16 编码的 .srt。要求用户自己先转格式，是把工具该做的事
    推给用户；ffmpeg 本来就能读这些。
    """
    with tempfile.TemporaryDirectory(prefix="la-subs-") as directory:
        source = Path(directory) / ("input" + (suffix or ".srt"))
        target = Path(directory) / "output.vtt"
        source.write_bytes(raw)
        command(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-f",
                "webvtt",
                str(target),
            ]
        )
        if not target.exists():
            raise MediaError("字幕转换失败，请检查文件是否完整")
        return target.read_text(encoding="utf-8", errors="replace")


def _decode_subtitle(raw):
    """按常见编码依次尝试解码字幕文本。

    字幕来源五花八门：国内课程常给 GBK/GB18030，旧工具导出 UTF-16（带 BOM），
    也有纯 UTF-8。ffmpeg 的 srt 解复用器对 UTF-16 并不可靠，所以编码这一层自己解，
    解不出来再退回 ffmpeg 转换。
    """
    candidates = ["utf-8-sig"]
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        candidates.append("utf-16")
    candidates.append("gb18030")
    for encoding in candidates:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def parse_captions(raw, duration, suffix=".srt"):
    suffix = (suffix or ".srt").lower()
    if not suffix.startswith("."):
        suffix = "." + suffix
    text = None
    if suffix in _TEXT_TIMED_SUBTITLES:
        text = _decode_subtitle(raw)
    if text is None:
        # 非 SRT/VTT（ass/ssa/ttml/sbv/sub…）交给 ffmpeg 先转 VTT 再按同一套规则解析。
        text = _subtitle_as_webvtt(raw, suffix)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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
