import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from learning_agent.config import Settings
from learning_agent.context import profile_id
from learning_agent.jobs import JobConflict, Jobs
from learning_agent.learning import Feedback, FeedbackError, LearningConflict, UserStateEdit
from learning_agent.media import (
    MEDIA_SUFFIXES,
    SUBTITLE_SUFFIXES,
    MediaError,
    asr_available,
    content_type,
    parse_captions,
)
from learning_agent.memory import ContextSettings, MemoryCreate, MemoryEdit, TopicCreate
from learning_agent.model_registry import status as model_status
from learning_agent.ocr import available as ocr_available
from learning_agent.pdf import PdfError
from learning_agent.provider import OpenAIProvider
from learning_agent.references import resolve_references
from learning_agent.retrieval import Retriever
from learning_agent.run_store import TERMINAL, RuntimeUnavailable
from learning_agent.runtime import RuntimeService
from learning_agent.store import Store
from learning_agent.uploads import Uploads

# 纯文本资料：UTF-8 解码后整段入库，格式本身不承担语义，没必要用白名单抠门。
TEXT_SUFFIXES = {
    ".txt", ".text", ".md", ".markdown", ".rst", ".tex", ".csv", ".tsv",
    ".json", ".log", ".yaml", ".yml",
}  # fmt: skip


class Text(BaseModel):
    content: str = Field(min_length=1, max_length=6000)

    @field_validator("content")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("内容不能为空")
        return text.strip()


class Focus(BaseModel):
    material_id: str
    page: int | None = Field(default=None, ge=1, le=200)
    start_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def position(self):
        if self.page is not None:
            if self.start_seconds is not None or self.end_seconds is not None:
                raise ValueError("请选择页码或媒体片段之一")
        elif (
            self.start_seconds is None
            or self.end_seconds is None
            or not 0 < self.end_seconds - self.start_seconds <= 120
        ):
            raise ValueError("媒体片段需要顺序正确的起止时间，最多120秒")
        return self


class Question(Text):
    focus: Focus | None = None
    feedback: Feedback | None = None


class RunQuestion(Question):
    client_request_id: str = Field(min_length=8, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")


class ResumeRequest(BaseModel):
    action_id: str = Field(min_length=8, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")
    expected_revision: int = Field(ge=0)
    acknowledge_unknown_request: bool = False


class Source(BaseModel):
    url: HttpUrl | None = Field(default=None, max_length=2048)
    start_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)


class UploadStart(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    size: int = Field(ge=1)
    last_modified: float | None = Field(default=None, ge=0, allow_inf_nan=False)


def create_app(settings=None, provider=None):
    settings = settings or Settings()
    store = Store(settings.data_root)
    retriever = Retriever(store, settings)
    jobs = Jobs(store)
    uploads = Uploads(store, jobs, settings)
    model = provider or OpenAIProvider(settings)
    runtime = RuntimeService(store, model, settings, retriever)

    @asynccontextmanager
    async def lifespan(app):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Learning Agent · 多模态学习讲解", lifespan=lifespan)
    app.state.store = store
    app.state.jobs = jobs
    app.state.runtime = runtime
    from learning_agent import access, accounts, survey, visitor
    from learning_agent.evaluation.web import install

    visitor.install(app)
    install(app, settings)
    # 账号与共享口令互斥：accounts 模式自己处理登录，口令只在 code 模式生效。
    if accounts.install(app, settings, store) == "code":
        access.install(app, settings)
    survey.install(app, store)

    @app.get("/healthz")
    async def healthz():
        # 不设口令时也应有健康检查；部署脚本和反向代理需要这个入口。
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")

    @app.exception_handler(MediaError)
    async def media_error(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(JobConflict)
    async def job_conflict(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(LookupError)
    async def missing(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(error)}, status_code=404)

    def owned(conversation_id, request):
        """只放行会话本人；``owner`` 为空表示公共会话（示例、迁移前历史、评测产物）。

        越权一律按"不存在"处理，不暴露 id 是否有效。
        """
        conversation = store.get_conversation(conversation_id)
        owner = conversation.get("owner") or ""
        if owner and owner != accounts.owner_key(request):
            raise LookupError("会话不存在")
        return conversation

    def owned_run(run_id, request):
        owned(store.run_conversation(run_id), request)
        return run_id

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return Path(__file__).with_name("index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    async def status():
        return {
            "model": settings.model,
            "protocol": settings.protocol,
            "configured": bool(settings.api_key.get_secret_value()) or provider is not None,
            "stream_answer": settings.stream_answer,
            "materials": ["txt", "md", "pdf", "audio", "video"],
            "vision": settings.enable_vision,
            "video": True,
            "asr_provider": settings.asr_provider,
            "asr_ready": (
                (settings.asr_provider == "local" and asr_available())
                or (
                    settings.asr_provider == "compatible"
                    and bool(settings.asr_api_key.get_secret_value())
                )
            ),
            "ocr_enabled": settings.ocr_enabled,
            "ocr_ready": settings.ocr_enabled and ocr_available(),
            "retrieval": settings.retrieval_mode,
            "embedding_provider": settings.embedding_provider,
            "capabilities": model_status(settings),
            "verification": "inspect_actual_run_results",
            "worker_online": jobs.online(),
            "runtime_online": runtime.task is not None
            and not runtime.task.done()
            and not runtime.stopping,
            "media_max_bytes": settings.media_max_bytes,
            "media_max_seconds": settings.media_max_seconds,
        }

    @app.get("/api/spaces")
    async def spaces():
        return store.spaces()

    @app.post("/api/spaces", status_code=201)
    async def new_space(payload: Text):
        if len(payload.content) > 120:
            raise HTTPException(422, "名称最多120字符")
        return store.space(payload.content)

    @app.get("/api/spaces/{space_id}/materials")
    async def materials(space_id: str):
        return store.materials(space_id)

    @app.get("/api/spaces/{space_id}/retrieval")
    async def retrieval_status(space_id: str):
        return await asyncio.to_thread(retriever.status, space_id)

    @app.post("/api/spaces/{space_id}/retrieval", status_code=202)
    async def prepare_index(space_id: str):
        store.require_space(space_id)
        if settings.embedding_provider == "disabled":
            raise HTTPException(503, "尚未配置向量服务，关键词检索仍可用")
        return jobs.enqueue("index", space_id, settings)

    @app.get("/api/spaces/{space_id}/jobs")
    async def list_jobs(space_id: str):
        return {"worker_online": jobs.online(), "jobs": jobs.list(space_id)}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        return {**jobs.get(job_id), "worker_online": jobs.online()}

    @app.post("/api/jobs/{job_id}/{action}")
    async def change_job(job_id: str, action: str):
        return jobs.action(job_id, action)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/outline")
    async def outline(space_id: str, material_id: str, start: int = 0):
        if start < 0:
            raise HTTPException(422, "目录偏移不能为负")
        return await asyncio.to_thread(retriever.index.outline, space_id, material_id, start, 30)

    @app.put("/api/spaces/{space_id}/materials/{material_id}/source")
    async def set_source(space_id: str, material_id: str, payload: Source):
        material = store.material(space_id, material_id)
        if payload.start_seconds and material["format"] not in {"audio", "video"}:
            raise HTTPException(422, "原始时间偏移仅适用于音视频")
        updated = store.set_source(space_id, material_id, payload.model_dump(mode="json"))
        # A different source rewrites every chunk of this material; rebuild the
        # index here so the space never sits on an empty index by accident.
        jobs.queue_index(space_id, settings)
        return updated

    @app.post("/api/spaces/{space_id}/materials", status_code=201)
    async def upload(space_id: str, file: UploadFile, response: Response):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in TEXT_SUFFIXES | {".pdf"} | MEDIA_SUFFIXES:
            raise HTTPException(422, "支持文字（txt/md/rst/tex/csv/json 等）、PDF 与常见音视频格式")
        if suffix in MEDIA_SUFFIXES:
            session = uploads.create(space_id, (file.filename or "media")[:200], file.size or 0)
            while block := await file.read(uploads.chunk_size):

                async def stream():
                    yield block

                session = await uploads.append(session["id"], session["received"], stream())
            item = await uploads.complete(session["id"])
            response.status_code = 202
            return item
        limit = 20_000_000 if suffix == ".pdf" else 2_000_000
        raw = await file.read(limit + 1)
        if len(raw) > limit:
            raise HTTPException(413, "PDF限制20 MB，文字资料限制2 MB")
        if suffix == ".pdf":
            try:
                item = await asyncio.to_thread(
                    store.add_pdf,
                    space_id,
                    (file.filename or "material.pdf")[:200],
                    raw,
                    settings.ocr_min_text_characters,
                )
            except PdfError as error:
                raise HTTPException(422, str(error)) from None
            if (
                settings.ocr_on_upload
                and settings.ocr_enabled
                and item["ocr_pending"]
                and ocr_available()
            ):
                # Opt-in only: recognising a whole scan on upload costs one
                # inference per page and nobody may ever read those pages.
                # Queued rather than inline, because a scanned book is hundreds
                # of pages; the default path recognises a page when it is read.
                try:
                    item["ocr_job"] = jobs.enqueue("ocr", space_id, settings, item["id"])
                except JobConflict:
                    pass
            return item
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(422, "请上传 UTF-8 编码的文字资料") from None
        if not text.strip():
            raise HTTPException(422, "资料内容不能为空")
        return store.add_material(space_id, (file.filename or "material.txt")[:200], text)

    @app.post("/api/spaces/{space_id}/uploads", status_code=201)
    async def start_upload(space_id: str, payload: UploadStart):
        return uploads.create(space_id, payload.filename, payload.size, payload.last_modified)

    @app.get("/api/uploads/{upload_id}")
    async def upload_status(upload_id: str):
        return uploads.get(upload_id)

    @app.patch("/api/uploads/{upload_id}")
    async def upload_chunk(upload_id: str, request: Request, upload_offset: int = Header(ge=0)):
        return await uploads.append(upload_id, upload_offset, request.stream())

    @app.post("/api/uploads/{upload_id}/complete", status_code=202)
    async def finish_upload(upload_id: str):
        return await uploads.complete(upload_id)

    @app.delete("/api/uploads/{upload_id}")
    async def cancel_upload(upload_id: str):
        return uploads.cancel(upload_id)

    @app.get("/api/spaces/{space_id}/materials/{material_id}")
    async def material(space_id: str, material_id: str):
        return store.material(space_id, material_id)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/media")
    async def media(space_id: str, material_id: str):
        return store.media(space_id, material_id)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/segments")
    async def segments(space_id: str, material_id: str):
        return store.segments(space_id, material_id)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/playback")
    async def playback(space_id: str, material_id: str):
        item = store.media(space_id, material_id)
        if item["state"] != "ready":
            raise HTTPException(409, item["error"] or "媒体仍在准备，请稍后刷新")
        path = store.root / item["preview_path"]
        # 按实际文件给类型：直接可播时 preview_path 就是原文件，可能是 webm/mov。
        return FileResponse(path, media_type=content_type(path))

    @app.post("/api/spaces/{space_id}/materials/{material_id}/prepare", status_code=202)
    async def retry_media(space_id: str, material_id: str):
        return jobs.enqueue("preview", space_id, settings, material_id)

    @app.post("/api/spaces/{space_id}/materials/{material_id}/captions")
    async def captions(space_id: str, material_id: str, file: UploadFile):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUBTITLE_SUFFIXES:
            raise HTTPException(422, "支持 SRT / VTT / ASS / SSA / TTML / SBV / SUB 等字幕格式")
        raw = await file.read(2_000_001)
        if len(raw) > 2_000_000:
            raise HTTPException(413, "字幕文件最多2 MB")
        item = store.media(space_id, material_id)
        # 非 SRT/VTT 由 ffmpeg 先转 VTT 再解析，不要求用户自己转格式。
        segments = parse_captions(raw, item["duration"], suffix)
        store.save_segments(
            space_id, material_id, segments, "uploaded:" + (file.filename or "captions")[:200]
        )
        jobs.queue_index(space_id, settings)
        return {"segment_count": len(segments), "transcript_state": "ready"}

    @app.post("/api/spaces/{space_id}/materials/{material_id}/ocr", status_code=202)
    async def recognize_pages(space_id: str, material_id: str):
        if store.material(space_id, material_id)["format"] != "pdf":
            raise HTTPException(422, "页面文字识别只适用于 PDF")
        if not settings.ocr_enabled:
            raise HTTPException(503, "OCR 已关闭；设置 LA_OCR_ENABLED=1 后重试")
        if not ocr_available():
            raise HTTPException(503, "未安装 OCR 依赖，请运行 uv sync --extra ocr")
        return jobs.enqueue("ocr", space_id, settings, material_id)

    @app.post("/api/spaces/{space_id}/materials/{material_id}/transcribe", status_code=202)
    async def transcribe(space_id: str, material_id: str):
        if settings.asr_provider == "disabled":
            raise HTTPException(503, "请配置语音识别，或先上传SRT/VTT字幕")
        # Content problems (no audio track) are reported before install problems.
        if not store.media(space_id, material_id)["has_audio"]:
            raise HTTPException(422, "该视频没有音轨，可查看画面或上传已有字幕")
        if settings.asr_provider == "local" and not asr_available():
            raise HTTPException(
                503, "未安装本地转写依赖，请运行 uv sync --extra asr；也可先上传SRT/VTT字幕"
            )
        return jobs.enqueue("transcribe", space_id, settings, material_id)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/pages")
    async def pages(space_id: str, material_id: str):
        return store.pages(space_id, material_id)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/pages/{page}")
    async def page_text(space_id: str, material_id: str, page: int):
        return store.page(space_id, material_id, page)

    @app.get("/api/spaces/{space_id}/materials/{material_id}/pages/{page}/image")
    async def page_image(space_id: str, material_id: str, page: int):
        try:
            image = await asyncio.to_thread(store.page_image, space_id, material_id, page)
        except PdfError as error:
            raise HTTPException(422, str(error)) from None
        return FileResponse(store.root / image["asset_path"], media_type="image/png")

    @app.get("/api/spaces/{space_id}/conversations")
    async def conversations(space_id: str, request: Request):
        return store.conversations(space_id, owner=accounts.owner_key(request))

    @app.post("/api/spaces/{space_id}/conversations", status_code=201)
    async def new_conversation(space_id: str, request: Request):
        return store.conversation(space_id, owner=accounts.owner_key(request))

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str, request: Request):
        owned(conversation_id, request)
        if not store.get_conversation(conversation_id).get("owner"):
            raise HTTPException(403, "公共示例会话不能删除")
        if store.list_runs(conversation_id, active=True, limit=1):
            raise HTTPException(409, "该会话还有进行中的问答，请先停止")
        store.delete_conversation(conversation_id)

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(conversation_id: str, request: Request):
        owned(conversation_id, request)
        return store.messages(conversation_id)

    def submit_question(conversation_id, payload, request_id):
        conv = store.get_conversation(conversation_id)
        focus = payload.focus.model_dump(exclude_none=True) if payload.focus else None
        prior = store.existing_submission(
            conversation_id,
            request_id,
            {
                "content": payload.content,
                "focus": focus,
                "feedback": payload.feedback.model_dump() if payload.feedback else None,
            },
        )
        if prior:
            return prior
        if focus:
            if "page" in focus:
                store.page(conv["space_id"], focus["material_id"], focus["page"])
            else:
                item = store.media(conv["space_id"], focus["material_id"])
                if focus["end_seconds"] > item["duration"]:
                    raise HTTPException(422, "片段超出媒体时长")
        if provider is None and not settings.api_key.get_secret_value():
            raise HTTPException(503, "请在本机 .env 中配置 LA_API_KEY；当前不会调用模型。")
        try:
            return runtime.submit(
                conversation_id,
                {
                    "content": payload.content,
                    "focus": focus,
                    "feedback": payload.feedback.model_dump() if payload.feedback else None,
                    "client_request_id": request_id,
                },
            )
        except FeedbackError as error:
            raise HTTPException(422, str(error)) from None
        except sqlite3.IntegrityError:
            raise HTTPException(409, "该会话正在回答，请等待完成后再发送。") from None

    @app.exception_handler(RuntimeUnavailable)
    async def runtime_unavailable(_request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.post("/api/conversations/{conversation_id}/messages")
    async def ask(
        conversation_id: str,
        payload: Question,
        request: Request,
        idempotency_key: str | None = Header(
            default=None, min_length=8, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$"
        ),
    ):
        owned(conversation_id, request)
        run_id = submit_question(conversation_id, payload, idempotency_key or str(uuid4()))
        return await runtime.wait_result(run_id)

    @app.post("/api/conversations/{conversation_id}/runs", status_code=202)
    async def submit(conversation_id: str, payload: RunQuestion, request: Request):
        owned(conversation_id, request)
        run_id = submit_question(conversation_id, payload, payload.client_request_id)
        return {
            "run_id": run_id,
            "state": store.public_run(run_id)["state"],
            "url": f"/api/runs/{run_id}",
            "events_url": f"/api/runs/{run_id}/events",
        }

    @app.get("/api/conversations/{conversation_id}/runs")
    async def conversation_runs(
        conversation_id: str,
        request: Request,
        status: str = Query(default="all", pattern="^(active|all)$"),
        before: int | None = Query(default=None, ge=1),
        limit: int = Query(default=20, ge=1, le=50),
    ):
        owned(conversation_id, request)
        return store.list_runs(conversation_id, status == "active", before, limit)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request):
        owned_run(run_id, request)
        store.cancel_run(run_id)
        runtime.wake.set()
        return store.public_run(run_id)

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    async def resume_run(run_id: str, payload: ResumeRequest, request: Request):
        owned_run(run_id, request)
        if runtime.task is None or runtime.task.done() or runtime.stopping:
            raise RuntimeUnavailable("问答调度器尚未就绪")
        result = store.resume_run(
            run_id,
            settings,
            payload.action_id,
            payload.expected_revision,
            payload.acknowledge_unknown_request,
        )
        runtime.wake.set()
        return result

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None),
    ):
        owned_run(run_id, request)
        try:
            cursor = int(last_event_id) if last_event_id is not None else after
            events = store.progress_events(run_id, cursor)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        if "text/event-stream" not in request.headers.get("accept", ""):
            return events

        async def stream():
            nonlocal cursor
            heartbeat = 0
            while not await request.is_disconnected():
                batch = store.progress_events(run_id, cursor)
                for event in batch:
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                current = store.public_run(run_id)
                if (
                    current["state"] in TERMINAL or current["state"] == "interrupted"
                ) and cursor >= current["last_event_seq"]:
                    return
                heartbeat += 1
                if heartbeat >= 40:
                    yield ": heartbeat\n\n"
                    heartbeat = 0
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/conversations/{conversation_id}/learning")
    async def learning_state(conversation_id: str, request: Request):
        owned(conversation_id, request)
        result = store.learning_state(conversation_id)
        summary = store.latest_summary_attempt(
            conversation_id, result["state"]["context_epoch"], profile_id(settings), ready=True
        )
        return {**result, "summary": summary}

    @app.exception_handler(LearningConflict)
    async def learning_conflict(_request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.get("/api/spaces/{space_id}/learning-topics")
    async def topics(space_id: str, request: Request):
        return store.learning_topics(space_id, accounts.owner_key(request))

    @app.post("/api/spaces/{space_id}/learning-topics", status_code=201)
    async def create_topic(space_id: str, payload: TopicCreate, request: Request):
        return store.create_learning_topic(
            space_id, payload.model_dump(), accounts.owner_key(request)
        )

    @app.get("/api/conversations/{conversation_id}/context-settings")
    async def context_settings(conversation_id: str, request: Request):
        owned(conversation_id, request)
        return store.context_settings(conversation_id)

    @app.patch("/api/conversations/{conversation_id}/context-settings")
    async def update_context_settings(
        conversation_id: str, payload: ContextSettings, request: Request
    ):
        owned(conversation_id, request)
        try:
            return store.edit_context_settings(conversation_id, payload.model_dump())
        except LearningConflict:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/spaces/{space_id}/memories")
    async def memories(space_id: str, topic_id: str, request: Request):
        return store.memories(space_id, topic_id, accounts.owner_key(request))

    @app.post("/api/spaces/{space_id}/memories", status_code=201)
    async def create_memory(space_id: str, payload: MemoryCreate, request: Request):
        try:
            return store.create_memory(space_id, payload.model_dump(), accounts.owner_key(request))
        except LearningConflict:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.patch("/api/spaces/{space_id}/memories/{memory_id}")
    async def update_memory(space_id: str, memory_id: str, payload: MemoryEdit, request: Request):
        try:
            return store.edit_memory(
                space_id, memory_id, payload.model_dump(), owner=accounts.owner_key(request)
            )
        except LearningConflict:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.delete("/api/spaces/{space_id}/memories/{memory_id}")
    async def forget_memory(space_id: str, memory_id: str, revision: int, request: Request):
        return store.edit_memory(
            space_id,
            memory_id,
            {"revision": revision},
            forget=True,
            owner=accounts.owner_key(request),
        )

    @app.patch("/api/conversations/{conversation_id}/learning")
    async def edit_learning(conversation_id: str, payload: UserStateEdit, request: Request):
        owned(conversation_id, request)
        try:
            return store.edit_learning_state(
                conversation_id, payload.model_dump(exclude_unset=True)
            )
        except LearningConflict as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/runs/{run_id}")
    async def run(run_id: str, request: Request):
        owned_run(run_id, request)
        result = store.public_run(run_id)
        answer, events = store.reference_context(run_id)
        space_id = store.get_conversation(result["conversation_id"])["space_id"]
        return {
            **result,
            "references": resolve_references(answer, events, space_id),
            "context": store.run_context(run_id),
        }

    @app.get("/api/runs/{run_id}/context")
    async def run_context(run_id: str, request: Request):
        owned_run(run_id, request)
        return store.run_context(run_id)

    return app
