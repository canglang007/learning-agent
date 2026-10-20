import pytest
from client_helpers import clients


@pytest.fixture(autouse=True)
def close_test_lifespans():
    yield
    for client in reversed(clients):
        client.close()
    clients.clear()
