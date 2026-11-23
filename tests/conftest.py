import os

import pytest
from client_helpers import clients

# 产品默认开完整检索（向量 + 精排）。测试不跟着下载 1.6 GB 模型，也不依赖网络：
# 在这里统一关掉，需要的用例自己显式传 embedding_provider / rerank_model——
# 传参优先于环境变量。默认值本身由 test_retrieval 里的断言守着。
os.environ.setdefault("LA_EMBEDDING_PROVIDER", "disabled")
os.environ.setdefault("LA_RERANK_MODEL", "")


@pytest.fixture(autouse=True)
def close_test_lifespans():
    yield
    for client in reversed(clients):
        client.close()
    clients.clear()
