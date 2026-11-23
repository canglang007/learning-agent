import os

import pytest
from client_helpers import clients

# 产品默认开完整检索（ollama bge-m3 向量 + 精排）。测试不依赖外部守护进程、也不下载
# 精排权重：在这里统一关掉，需要的用例自己显式传 embedding_provider / rerank_model——
# 传参优先于环境变量。默认值本身由 test_retrieval 里的断言守着，不会被悄悄改回去。
os.environ.setdefault("LA_EMBEDDING_PROVIDER", "disabled")
os.environ.setdefault("LA_RERANK_MODEL", "")


@pytest.fixture(autouse=True)
def close_test_lifespans():
    yield
    for client in reversed(clients):
        client.close()
    clients.clear()
