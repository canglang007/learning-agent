import json

import pytest
from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.references import resolve_references
from learning_agent.retrieval import CourseIndex, EmbeddingError, Retriever, bm25, fuse
from learning_agent.store import Store
from learning_agent.tools import MaterialTools


class FakeEmbedder:
    key = "test-encoder"

    def __init__(self):
        self.calls = []
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("offline")
        self.calls.append(texts)
        return [[1.0, 0.0] if "normalization" in t or "归一化" in t else [0.0, 1.0] for t in texts]


def test_multilingual_rank_fusion_cache_restart_and_explicit_fallback(tmp_path):
    store = Store(tmp_path)
    space = store.space("course")["id"]
    relevant = store.add_material(space, "lecture.md", "normalization preserves residual path")
    store.add_material(space, "other.md", "weather example temperature")
    embedder = FakeEmbedder()
    config = Settings(_env_file=None, embedding_provider="ollama")
    retriever = Retriever(store, config, embedder)
    assert retriever.search(space, "归一化")["dense_status"] == "index_incomplete"
    retriever.prepare(space)
    assert len(embedder.calls) == 1
    restarted = Retriever(Store(tmp_path), config, embedder)
    restarted.prepare(space)
    assert len(embedder.calls) == 1  # persisted document vectors; no rebuilding after restart
    result = restarted.search(space, "归一化")
    assert result["retrieval"] == "hybrid_rrf"
    assert result["hits"][0]["id"] == relevant["id"]
    assert result["hits"][0]["bm25_rank"] is None
    embedder.fail = True
    result = restarted.search(space, "normalization")
    assert result["retrieval"] == "bm25" and result["dense_status"] == "offline"
    assert result["hits"][0]["id"] == relevant["id"]
    assert bm25("课程学习", ["课程学习", "unrelated"])[0] > 0
    order, _ = fuse([0, 1], [1, 2])
    assert order[0] == 1


def test_neighbors_navigation_scope_and_source_changes_invalidate_vectors(tmp_path):
    store = Store(tmp_path)
    space = store.space("course")["id"]
    other = store.space("private")["id"]
    material = store.add_material(
        space, "notes.md", "# definition\nnormalization\n# example\nexample\n# caveat\ncondition"
    )
    fake = FakeEmbedder()
    config = Settings(_env_file=None, embedding_provider="ollama")
    retriever = Retriever(store, config, fake)
    retriever.prepare(space)
    chunks = retriever.index.chunks(space)
    context = retriever.index.context(space, chunks[1]["chunk_id"], 1, 1)
    assert len(context["chunks"]) == 3
    with pytest.raises(LookupError):
        retriever.index.context(other, chunks[1]["chunk_id"], 1, 1)
    nav = retriever.index.outline(space, material["id"], 0, 1)
    assert nav["next_start"] == 1 and len(nav["entries"]) == 1
    store.set_source(
        space, material["id"], {"url": "https://example.org/lecture", "start_seconds": 0}
    )
    assert retriever.status(space)["embedded"] == 0
    with pytest.raises(LookupError):
        retriever.index.context(space, chunks[1]["chunk_id"], 1, 1)
    # A source update regenerates IDs; fresh searches are safe after invalidation.
    assert retriever.search(space, "normalization")["hits"]
    tools = MaterialTools(store, other)
    assert (
        tools.execute("material_outline", {"material_id": material["id"]})["error"]
        == "material_not_found"
    )
    assert (
        tools.execute("read_context", {"chunk_id": chunks[0]["chunk_id"]})["error"]
        == "material_not_found"
    )


def test_legacy_backfill_caption_replacement_and_original_time_mapping(tmp_path):
    store = Store(tmp_path)
    space = store.space("course")["id"]
    # Media metadata fixture isolates indexing from the already tested ffmpeg pipeline.
    with store.connect() as db:
        db.execute(
            "INSERT INTO materials(id,space_id,name,text,format) VALUES ('v',?,'course.mp4','','video')",
            (space,),
        )
        db.execute(
            "INSERT INTO media(material_id,metadata,state) VALUES ('v',?,'ready')",
            (json.dumps({"duration": 120}),),
        )
    store.save_segments(
        space, "v", [{"start": 10, "end": 20, "text": "old definition"}], "uploaded:old.vtt"
    )
    store.set_source(space, "v", {"url": "https://example.org/video", "start_seconds": 2700})
    index = CourseIndex(store)
    first = index.chunks(space)[0]
    assert first["original_start_seconds"] == 2710 and first["start_seconds"] == 10
    store.save_segments(
        space, "v", [{"start": 30, "end": 40, "text": "corrected definition"}], "uploaded:new.vtt"
    )
    second = index.chunks(space)[0]
    assert second["chunk_id"] != first["chunk_id"] and second["original_start_seconds"] == 2730
    assert "old definition" not in second["excerpt"]
    refs = resolve_references(
        "[course.mp4:00:30-00:40]", [{"type": "tool", "observation": {"chunks": [second]}}], space
    )
    assert refs[0]["position_verified"]


def test_index_api_navigation_source_validation_and_bounded_chunks(tmp_path):
    settings = Settings(_env_file=None, data_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        space = client.post("/api/spaces", json={"content": "course"}).json()["id"]
        m = client.post(
            f"/api/spaces/{space}/materials", files={"file": ("long.md", ("x" * 4000).encode())}
        ).json()["id"]
        base = f"/api/spaces/{space}"
        assert client.get(base + "/retrieval").json()["chunks"] == 4
        assert client.post(base + "/retrieval").status_code == 503
        assert client.get(base + "/retrieval").json()["provider"] == "disabled"
        assert client.get(base + f"/materials/{m}/outline?start=-1").status_code == 422
        assert (
            client.put(
                base + f"/materials/{m}/source", json={"url": "file:///etc/passwd"}
            ).status_code
            == 422
        )
        assert (
            client.put(base + f"/materials/{m}/source", json={"start_seconds": 30}).status_code
            == 422
        )


def test_compatible_embedding_order_validation_and_secret_free_errors(monkeypatch):
    import httpx

    from learning_agent import retrieval

    real_client = httpx.Client
    settings = Settings(
        _env_file=None,
        embedding_provider="compatible",
        embedding_base_url="https://example.org/v1",
        embedding_api_key="private-test-key",
    )

    def respond(request):
        assert request.url.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer private-test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 4.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            },
        )

    monkeypatch.setattr(
        retrieval.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )
    assert retrieval.Embedder(settings).embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]

    def fail(request):
        return httpx.Response(401, text="private-test-key and confidential upstream detail")

    monkeypatch.setattr(
        retrieval.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(fail), **kw),
    )
    with pytest.raises(EmbeddingError) as error:
        retrieval.Embedder(settings).embed(["private text"])
    assert "private" not in str(error.value) and "upstream" not in str(error.value)


def test_explicit_course_identifier_scopes_documents_without_title_boost():
    from learning_agent.retrieval import course_scope

    chunks = [
        {"id": "p", "name": "cs336-architecture.pdf", "excerpt": "CS336 course cover"},
        {"id": "v", "name": "cs336-first45min.mp4", "excerpt": "rotary embeddings"},
        {"id": "other", "name": "ntu-agent.pdf", "excerpt": "CS336 mentioned in another course"},
    ]
    selected, query, ids = course_scope("CS336旋转位置编码如何起作用？", chunks)
    assert ids == ["p", "v"] and [c["id"] for c in selected] == ["p", "v"]
    assert "cs336" not in query.casefold()
    selected, query, ids = course_scope("位置编码", chunks)
    assert selected == chunks and ids == [] and query == "位置编码"
    # A course-directory query remains searchable instead of becoming an empty vector input.
    assert course_scope("CS336", chunks)[1] == "CS336"
