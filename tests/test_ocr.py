import io

from client_helpers import TestClient, close_for
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from test_jobs import Batches
from test_learning import Scripted

import learning_agent.app as app_module
import learning_agent.ocr as ocr_module
from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.ocr import available
from learning_agent.retrieval import Retriever
from learning_agent.store import Store
from learning_agent.tools import MaterialTools
from learning_agent.worker import Worker


def scanned_pdf():
    """Page 1 has a real text layer; page 2 is a raster only (a scan)."""
    buffer = io.BytesIO()
    doc = canvas.Canvas(buffer, pagesize=(500, 400))
    doc.drawString(30, 350, "Page one: the chain rule and its conditions")
    doc.showPage()
    image = Image.new("RGB", (400, 250), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(30, 20), (30, 220), (380, 220)], fill="black", width=3)
    draw.text((100, 20), "Loss against step", fill="black")
    doc.drawImage(ImageReader(image), 40, 80, width=400, height=250)
    doc.save()
    return buffer.getvalue()


def prepared(root, **settings):
    config = Settings(_env_file=None, data_root=root, **settings)
    close_for(root)
    client = TestClient(create_app(config, Scripted([])))
    space = client.post("/api/spaces", json={"content": "scan study"}).json()["id"]
    uploaded = client.post(
        f"/api/spaces/{space}/materials",
        files={"file": ("scan.pdf", scanned_pdf(), "application/pdf")},
    )
    assert uploaded.status_code == 201
    return client, space, uploaded.json(), config


class Engine:
    """Stand-in for the OCR engine; counts how many pages were recognised."""

    def __init__(self, text="recognised: y = wx + b"):
        self.calls = 0
        self.text = text

    def __call__(self, png):
        self.calls += 1
        return {"text": self.text, "lines": 1, "mean_score": 0.91}


def test_pages_record_their_text_source_and_upload_stays_cheap(tmp_path, monkeypatch):
    engine = Engine()
    monkeypatch.setattr(ocr_module, "available", lambda: True)
    monkeypatch.setattr(ocr_module, "recognize", engine)
    client, space, uploaded, config = prepared(tmp_path)

    store = Store(config.data_root)
    pages = client.get(f"/api/spaces/{space}/materials/{uploaded['id']}/pages").json()
    assert [page["text_source"] for page in pages] == ["text", "none"]
    assert uploaded["ocr_pending"] == 1
    # Default is on demand: nothing is recognised just because a file arrived.
    assert "ocr_job" not in uploaded
    assert engine.calls == 0
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs WHERE kind='ocr'").fetchone()[0] == 0
    assert store.materials(space)[0]["pages_without_text"] == 1


def test_material_outline_shows_which_pages_are_scans(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_module, "available", lambda: True)
    _, space, uploaded, config = prepared(tmp_path)
    store = Store(config.data_root)
    outline = Retriever(store, config).index.outline(space, uploaded["id"], 0, 30)
    assert [entry["text_source"] for entry in outline["entries"]] == ["text", "none"]
    assert outline["entries"][1]["chunk_id"] is None


def test_ocr_on_upload_is_opt_in_and_confined_to_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr_module,
        "recognize",
        lambda png: {"text": "recognised: y = wx + b", "lines": 1, "mean_score": 0.91},
    )
    _, space, uploaded, config = prepared(tmp_path, ocr_on_upload=True)
    assert uploaded["ocr_job"]["kind"] == "ocr"
    assert uploaded["ocr_job"]["state"] == "queued"

    store = Store(config.data_root)
    before = store.material(space, uploaded["id"])["revision"]
    with Worker(config) as worker:
        worker.drain()

    page = store.page(space, uploaded["id"], 2)
    assert page["text_source"] == "ocr"
    assert "y = wx + b" in page["text"]
    after = store.material(space, uploaded["id"])
    assert after["revision"] == before + 1
    assert "y = wx + b" in after["text"]


def test_reading_a_scanned_page_recognises_that_page_only(tmp_path, monkeypatch):
    engine = Engine()
    monkeypatch.setattr(ocr_module, "available", lambda: True)
    monkeypatch.setattr(ocr_module, "recognize", engine)
    _, space, uploaded, config = prepared(tmp_path)
    store = Store(config.data_root)
    before = store.material(space, uploaded["id"])
    chunks_before = len(Retriever(store, config).index.chunks(space))

    tools = MaterialTools(store, space)
    observed = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert observed["text_source"] == "ocr"
    assert "y = wx + b" in observed["text"]
    assert "检索索引" in observed["detail"]
    assert engine.calls == 1

    # The page is readable, but the index was not touched: no revision bump, no
    # chunk, no queued job. Publishing stays an explicit action.
    after = store.material(space, uploaded["id"])
    assert after["revision"] == before["revision"]
    assert after["text"] == before["text"]
    assert len(Retriever(store, config).index.chunks(space)) == chunks_before
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs WHERE kind='ocr'").fetchone()[0] == 0

    # Recognising the same page again is a cache hit, not a second inference.
    again = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert again["text_source"] == "ocr" and engine.calls == 1


def test_figure_only_page_is_marked_blank_and_never_retried(tmp_path, monkeypatch):
    engine = Engine(text="   ")
    monkeypatch.setattr(ocr_module, "available", lambda: True)
    monkeypatch.setattr(ocr_module, "recognize", engine)
    _, space, uploaded, config = prepared(tmp_path)
    store = Store(config.data_root)
    revision = store.material(space, uploaded["id"])["revision"]

    tools = MaterialTools(store, space)
    first = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert first["text_source"] == "blank"
    assert first["has_text"] is False
    assert "纯图页" in first["detail"]

    second = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert second["text_source"] == "blank"
    assert engine.calls == 1
    # A page with nothing to publish leaves the document untouched.
    assert store.material(space, uploaded["id"])["revision"] == revision


def test_explicit_recognition_publishes_and_rebuilds_the_index_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr_module,
        "recognize",
        lambda png: {"text": "recognised: y = wx + b", "lines": 1, "mean_score": 0.91},
    )
    client, space, uploaded, config = prepared(tmp_path, embedding_provider="ollama")
    store = Store(config.data_root)
    jobs = Jobs(store)
    before = store.material(space, uploaded["id"])["revision"]
    assert "ocr_job" not in uploaded

    item = client.post(f"/api/spaces/{space}/materials/{uploaded['id']}/ocr").json()
    assert item["kind"] == "ocr"
    with Worker(config, Retriever(store, config, Batches())) as worker:
        worker.drain()

    assert store.page(space, uploaded["id"], 2)["text_source"] == "ocr"
    published = store.material(space, uploaded["id"])
    assert published["revision"] == before + 1
    assert "y = wx + b" in published["text"]

    # The text became searchable without anyone clicking "build index": the OCR
    # job queues that rebuild itself, and the chunks carry the recognised text.
    chunks = Retriever(store, config).index.chunks(space)
    assert any("y = wx + b" in chunk["excerpt"] for chunk in chunks)
    with store.connect() as db:
        index_jobs = db.execute("SELECT state FROM jobs WHERE kind='index'").fetchall()
        assert [row[0] for row in index_jobs] == ["succeeded"]
        assert db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] > 0
    assert jobs.version("ocr", space, uploaded["id"]) == "pages:2"


def test_ocr_job_input_version_survives_its_own_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr_module,
        "recognize",
        lambda png: {"text": "second page text", "lines": 1, "mean_score": 0.9},
    )
    client, space, uploaded, config = prepared(tmp_path, ocr_on_upload=True)
    store = Store(config.data_root)
    jobs = Jobs(store)
    before = jobs.get(uploaded["ocr_job"]["id"])["input_version"]
    with Worker(config) as worker:
        worker.drain()
    # OCR changes page text and the material revision. Keying its input on either
    # would abort the job on its own first write; the page set is the real input.
    assert jobs.version("ocr", space, uploaded["id"]) == before == "pages:2"
    assert client.get("/api/status").status_code == 200


def test_disabled_switch_skips_queue_and_refuses_recognition(tmp_path, monkeypatch):
    engine = Engine()
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr_module, "available", lambda: True)
    monkeypatch.setattr(ocr_module, "recognize", engine)
    client, space, uploaded, config = prepared(tmp_path, ocr_enabled=False, ocr_on_upload=True)
    assert "ocr_job" not in uploaded

    store = Store(config.data_root)
    tools = MaterialTools(store, space, retriever=Retriever(store, config))
    observed = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert observed["text_source"] == "none" and engine.calls == 0

    response = client.post(f"/api/spaces/{space}/materials/{uploaded['id']}/ocr")
    assert response.status_code == 503
    assert "LA_OCR_ENABLED" in response.json()["detail"]


def test_missing_engine_keeps_the_page_honestly_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: False)
    monkeypatch.setattr(ocr_module, "available", lambda: False)
    client, space, uploaded, config = prepared(tmp_path, ocr_on_upload=True)
    assert "ocr_job" not in uploaded

    store = Store(config.data_root)
    tools = MaterialTools(store, space, retriever=Retriever(store, config))
    observed = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert observed["text_source"] == "none"
    assert observed["has_text"] is False

    response = client.post(f"/api/spaces/{space}/materials/{uploaded['id']}/ocr")
    assert response.status_code == 503
    assert "uv sync --extra ocr" in response.json()["detail"]


def test_recognition_failure_is_reported_not_invented(tmp_path, monkeypatch):
    def broken(png):
        raise ocr_module.OcrError("OCR 识别失败，请检查页面图像与可用内存")

    monkeypatch.setattr(ocr_module, "available", lambda: True)
    monkeypatch.setattr(ocr_module, "recognize", broken)
    _, space, uploaded, config = prepared(tmp_path)
    store = Store(config.data_root)
    tools = MaterialTools(store, space)
    observed = tools.execute("read_page", {"material_id": uploaded["id"], "page": 2})
    assert observed["text_source"] == "none" and observed["has_text"] is False
    assert "按需识别失败" in observed["detail"]


def test_ocr_only_applies_to_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    config = Settings(_env_file=None, data_root=tmp_path)
    close_for(tmp_path)
    client = TestClient(create_app(config, Scripted([])))
    space = client.post("/api/spaces", json={"content": "text only"}).json()["id"]
    material = client.post(
        f"/api/spaces/{space}/materials", files={"file": ("notes.md", b"# Title\nbody")}
    ).json()["id"]
    response = client.post(f"/api/spaces/{space}/materials/{material}/ocr")
    assert response.status_code == 422


def test_status_reports_asr_and_ocr_readiness(tmp_path):
    import importlib.util

    config = Settings(_env_file=None, data_root=tmp_path)
    close_for(tmp_path)
    client = TestClient(create_app(config, Scripted([])))
    status = client.get("/api/status").json()
    assert status["asr_provider"] == "local"
    assert status["asr_ready"] == (importlib.util.find_spec("faster_whisper") is not None)
    assert status["ocr_enabled"] is True
    assert status["ocr_ready"] == available()


def test_default_transcription_provider_is_local(tmp_path):
    assert Settings(_env_file=None, data_root=tmp_path).asr_provider == "local"
