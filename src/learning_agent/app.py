import asyncio
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

from learning_agent.agent import run_turn
from learning_agent.config import Settings
from learning_agent.pdf import PdfError
from learning_agent.provider import OpenAIProvider
from learning_agent.store import Store


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
    page: int = Field(ge=1, le=200)


class Question(Text):
    focus: Focus | None = None


def create_app(settings=None, provider=None):
    settings = settings or Settings()
    store = Store(settings.data_root)
    model = provider or OpenAIProvider(settings)
    app = FastAPI(title="Learning Agent · 资料与页面讲解")

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
            "materials": ["txt", "md", "pdf"],
            "vision": settings.enable_vision,
            "video": False,
            "retrieval": "lexical_baseline",
            "verification": "inspect_actual_run_results",
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

    @app.post("/api/spaces/{space_id}/materials", status_code=201)
    async def upload(space_id: str, file: UploadFile):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            raise HTTPException(422, "支持 UTF-8 TXT/Markdown 和 PDF。影音工具后续加入。")
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

    @app.get("/api/spaces/{space_id}/materials/{material_id}")
    async def material(space_id: str, material_id: str):
        return store.material(space_id, material_id)

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
        focus = payload.focus.model_dump() if payload.focus else None
        if focus:
            store.page(conv["space_id"], focus["material_id"], focus["page"])
        if provider is None and not settings.api_key.get_secret_value():
            raise HTTPException(503, "请在本机 .env 中配置 LA_API_KEY；当前不会调用模型。")
        try:
            return await run_turn(store, model, settings, conversation_id, payload.content, focus)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "该会话正在回答，请等待完成后再发送。") from None

    @app.get("/api/runs/{run_id}")
    async def run(run_id: str):
        return store.run(run_id)

    return app
