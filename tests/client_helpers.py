"""Tests explicitly start lifespan and close it, including clients created in helpers."""

from fastapi.testclient import TestClient as BaseClient

clients = []


class TestClient(BaseClient):
    __test__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = False
        clients.append(self)
        self.__enter__()

    def __enter__(self):
        if not self.started:
            super().__enter__()
            self.started = True
        return self

    def __exit__(self, *args):
        if self.started:
            super().__exit__(*args)
            self.started = False

    def close(self):
        self.__exit__(None, None, None)
        super().close()


def close_for(root):
    """Explicitly stop an old application before testing new settings/restart."""
    for client in clients:
        if client.app.state.store.root == root:
            client.close()
