import asyncio
import sqlite3
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from learning_agent.agent import run_turn
from learning_agent.config import Settings
from learning_agent.context import profile_id
from learning_agent.jobs import JobConflict, Jobs
from learning_agent.learning import Feedback, FeedbackError, LearningConflict, UserStateEdit
from learning_agent.media import MEDIA_SUFFIXES, MediaError, parse_captions
from learning_agent.memory import ContextSettings, MemoryCreate, MemoryEdit, TopicCreate
from learning_agent.pdf import PdfError
from learning_agent.provider import OpenAIProvider
from learning_agent.references import resolve_references
from learning_agent.retrieval import Retriever
from learning_agent.store import Store
from learning_agent.uploads import Uploads


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
    app = FastAPI(title="Learning Agent · 多模态学习讲解")
    app.state.store = store
    app.state.jobs = jobs
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

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return Path(__file__).with_name("index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    async def status():
        return {
            "model": settings.model,
            "protocol": settings.protocol,
            "configured": bool(settings.api_key.get_secret_value()) or provider is not None,
            "materials": ["txt", "md", "pdf", "audio", "video"],
            "vision": settings.enable_vision,
            "video": True,
            "asr_provider": settings.asr_provider,
            "retrieval": settings.retrieval_mode,
            "embedding_provider": settings.embedding_provider,
            "verification": "inspect_actual_run_results",
            "worker_online": jobs.online(),
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
        return store.set_source(space_id, material_id, payload.model_dump(mode="json"))

    @app.post("/api/spaces/{space_id}/materials", status_code=201)
    async def upload(space_id: str, file: UploadFile, response: Response):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"} | MEDIA_SUFFIXES:
            raise HTTPException(422, "支持文字、PDF、常用音频/视频格式")
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
                return await asyncio.to_thread(
                    store.add_pdf, space_id, (file.filename or "material.pdf")[:200], raw
                )
            except PdfError as error:
                raise HTTPException(422, str(error)) from None
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
        return FileResponse(
            store.root / item["preview_path"],
            media_type="video/mp4" if item["has_video"] else "audio/mp4",
        )

    @app.post("/api/spaces/{space_id}/materials/{material_id}/prepare", status_code=202)
    async def retry_media(space_id: str, material_id: str):
        return jobs.enqueue("preview", space_id, settings, material_id)

    @app.post("/api/spaces/{space_id}/materials/{material_id}/captions")
    async def captions(space_id: str, material_id: str, file: UploadFile):
        if Path(file.filename or "").suffix.lower() not in {".srt", ".vtt"}:
            raise HTTPException(422, "请上传UTF-8 SRT或VTT字幕")
        raw = await file.read(2_000_001)
        if len(raw) > 2_000_000:
            raise HTTPException(413, "字幕文件最多2 MB")
        item = store.media(space_id, material_id)
        segments = parse_captions(raw, item["duration"])
        store.save_segments(
            space_id, material_id, segments, "uploaded:" + (file.filename or "captions")[:200]
        )
        return {"segment_count": len(segments), "transcript_state": "ready"}

    @app.post("/api/spaces/{space_id}/materials/{material_id}/transcribe", status_code=202)
    async def transcribe(space_id: str, material_id: str):
        if settings.asr_provider == "disabled":
            raise HTTPException(503, "请配置语音识别，或先上传SRT/VTT字幕")
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
    async def conversations(space_id: str):
        return store.conversations(space_id)

    @app.post("/api/spaces/{space_id}/conversations", status_code=201)
    async def new_conversation(space_id: str):
        return store.conversation(space_id)

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(conversation_id: str):
        return store.messages(conversation_id)

    @app.post("/api/conversations/{conversation_id}/messages")
    async def ask(conversation_id: str, payload: Question):
        conv = store.get_conversation(conversation_id)
        focus = payload.focus.model_dump(exclude_none=True) if payload.focus else None
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
            return await run_turn(
                store,
                model,
                settings,
                conversation_id,
                payload.content,
                focus,
                retriever,
                payload.feedback.model_dump() if payload.feedback else None,
            )
        except FeedbackError as error:
            raise HTTPException(422, str(error)) from None
        except sqlite3.IntegrityError:
            raise HTTPException(409, "该会话正在回答，请等待完成后再发送。") from None

    @app.get("/api/conversations/{conversation_id}/learning")
    async def learning_state(conversation_id: str):
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
    async def topics(space_id: str):
        return store.learning_topics(space_id)

    @app.post("/api/spaces/{space_id}/learning-topics", status_code=201)
    async def create_topic(space_id: str, payload: TopicCreate):
        return store.create_learning_topic(space_id, payload.model_dump())

    @app.get("/api/conversations/{conversation_id}/context-settings")
    async def context_settings(conversation_id: str):
        return store.context_settings(conversation_id)

    @app.patch("/api/conversations/{conversation_id}/context-settings")
    async def update_context_settings(conversation_id: str, payload: ContextSettings):
        try:
            return store.edit_context_settings(conversation_id, payload.model_dump())
        except LearningConflict:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/spaces/{space_id}/memories")
    async def memories(space_id: str, topic_id: str):
        return store.memories(space_id, topic_id)

    @app.post("/api/spaces/{space_id}/memories", status_code=201)
    async def create_memory(space_id: str, payload: MemoryCreate):
        try:
            return store.create_memory(space_id, payload.model_dump())
        except LearningConflict:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.patch("/api/spaces/{space_id}/memories/{memory_id}")
    async def update_memory(space_id: str, memory_id: str, payload: MemoryEdit):
        try:
            return store.edit_memory(space_id, memory_id, payload.model_dump())
        except LearningConflict:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.delete("/api/spaces/{space_id}/memories/{memory_id}")
    async def forget_memory(space_id: str, memory_id: str, revision: int):
        return store.edit_memory(space_id, memory_id, {"revision": revision}, forget=True)

    @app.patch("/api/conversations/{conversation_id}/learning")
    async def edit_learning(conversation_id: str, payload: UserStateEdit):
        try:
            return store.edit_learning_state(
                conversation_id, payload.model_dump(exclude_unset=True)
            )
        except LearningConflict as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/runs/{run_id}")
    async def run(run_id: str):
        result = store.run(run_id)
        answer, events = store.reference_context(run_id)
        space_id = store.get_conversation(result["conversation_id"])["space_id"]
        return {
            **result,
            "references": resolve_references(answer, events, space_id),
            "context": store.run_context(run_id),
        }

    @app.get("/api/runs/{run_id}/context")
    async def run_context(run_id: str):
        return store.run_context(run_id)

    return app
