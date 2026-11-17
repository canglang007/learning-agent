import asyncio
import json
import sys

import pytest

from learning_agent.config import Settings
from learning_agent.retrieval import Embedder, EmbeddingError, Retriever
from learning_agent.store import Store


class FakeEngine:
    """替下 fastembed：只关心接口契约，不下载任何模型。"""

    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(list(texts))
        # 故意不归一化，交给 Embedder 统一处理
        return [[3.0, 4.0] for _ in texts]


def local_settings(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        data_root=tmp_path,
        retrieval_mode="hybrid",
        embedding_provider="local",
        embedding_model="test/model",
        **overrides,
    )


def test_local_key_ignores_http_fields(tmp_path):
    """换 base_url 不该让本地向量失效重建。"""
    first = Embedder(local_settings(tmp_path))
    second = Embedder(local_settings(tmp_path, embedding_base_url="http://elsewhere:1234"))
    assert first.key == second.key == json.dumps(["local", "test/model"])
    remote = Embedder(Settings(_env_file=None, embedding_provider="ollama"))
    assert remote.key != first.key


def test_local_embed_normalizes_and_keeps_order(tmp_path):
    embedder = Embedder(local_settings(tmp_path))
    engine = FakeEngine()
    embedder._engine = engine
    vectors = embedder.embed(["甲乙", "丙丁"])
    assert engine.batches == [["甲乙", "丙丁"]]
    assert vectors == [[0.6, 0.8], [0.6, 0.8]]


def test_local_async_path_uses_thread(tmp_path):
    embedder = Embedder(local_settings(tmp_path))
    embedder._engine = FakeEngine()
    assert asyncio.run(embedder.embed_async(["甲"])) == [[0.6, 0.8]]


def test_missing_extra_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    embedder = Embedder(local_settings(tmp_path))
    with pytest.raises(EmbeddingError) as error:
        embedder.embed(["甲"])
    # 报错要指向可执行的修复动作，而不是丢一个裸 ImportError
    assert str(error.value) == "embedding_local_unavailable:uv_sync_extra_embed"


def test_local_provider_marks_dense_ready(tmp_path):
    store = Store(tmp_path)
    space = store.space("course")["id"]
    store.add_material(space, "lecture.md", "normalization preserves the residual path")
    settings = local_settings(tmp_path)
    retriever = Retriever(store, settings, Embedder(settings))
    retriever.embedder._engine = FakeEngine()
    retriever.prepare(space)
    status = retriever.status(space)
    assert status["provider"] == "local" and status["dense_ready"] is True
    result = retriever.search(space, "归一化")
    assert result["retrieval"] == "hybrid_rrf" and result["dense_status"] == "ready"


def test_disabled_provider_stays_disabled(tmp_path):
    settings = Settings(_env_file=None, data_root=tmp_path, embedding_provider="disabled")
    embedder = Embedder(settings)
    with pytest.raises(EmbeddingError) as error:
        embedder.embed(["甲"])
    assert str(error.value) == "embedding_disabled"
