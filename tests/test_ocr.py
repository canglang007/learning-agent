import io

from client_helpers import TestClient, close_for
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
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


def test_pages_record_whether_text_came_from_the_file_or_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: False)
    client, space, uploaded, _ = prepared(tmp_path)
    pages = client.get(f"/api/spaces/{space}/materials/{uploaded['id']}/pages").json()
    assert [page["text_source"] for page in pages] == ["text", "none"]
    assert uploaded["ocr_pending"] == 1
    # Without the optional engine nothing is queued, and the page stays honestly empty.
    assert "ocr_job" not in uploaded


def test_upload_queues_ocr_and_the_job_rewrites_pages_and_index(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr_module,
        "recognize",
        lambda png: {"text": "recognised: y = wx + b", "lines": 1, "mean_score": 0.91},
    )
    _, space, uploaded, config = prepared(tmp_path)
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

    # The regenerated index now contains the recognised text.
    chunks = Retriever(store, config).index.chunks(space)
    assert any("y = wx + b" in chunk["excerpt"] for chunk in chunks)

    observed = MaterialTools(store, space).execute(
        "read_page", {"material_id": uploaded["id"], "page": 2}
    )
    assert observed["text_source"] == "ocr"
    assert "OCR" in observed["detail"]


def test_ocr_job_input_version_survives_its_own_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr_module,
        "recognize",
        lambda png: {"text": "second page text", "lines": 1, "mean_score": 0.9},
    )
    _, space, uploaded, config = prepared(tmp_path)
    store = Store(config.data_root)
    jobs = Jobs(store)
    before = jobs.get(uploaded["ocr_job"]["id"])["input_version"]
    with Worker(config) as worker:
        worker.drain()
    # OCR changes page text and the material revision. Keying its input on either
    # would abort the job on its own first write; the page set is the real input.
    assert jobs.version("ocr", space, uploaded["id"]) == before == "pages:2"


def test_ocr_endpoint_explains_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: False)
    client, space, uploaded, _ = prepared(tmp_path)
    response = client.post(f"/api/spaces/{space}/materials/{uploaded['id']}/ocr")
    assert response.status_code == 503
    assert "uv sync --extra ocr" in response.json()["detail"]


def test_disabled_ocr_switch_skips_queue_and_refuses_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ocr_available", lambda: True)
    client, space, uploaded, _ = prepared(tmp_path, ocr_enabled=False)
    assert "ocr_job" not in uploaded
    response = client.post(f"/api/spaces/{space}/materials/{uploaded['id']}/ocr")
    assert response.status_code == 503
    assert "LA_OCR_ENABLED" in response.json()["detail"]


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
