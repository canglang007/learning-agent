"""Tests explicitly start lifespan and close it, including clients created in helpers."""

from fastapi.testclient import TestClient as BaseClient

clients = []


class TestClient(BaseClient):
    __test__ = False

    def __init__(self, *args, **kwargs):
        carry = kwargs.pop("carry_visitor", True)
        super().__init__(*args, **kwargs)
        self.started = False
        # 同一个数据目录上的新 client 代表"服务重启后同一台浏览器再打开"，
        # 访客标识存在 cookie 里，应当延续；否则会被会话隔离当成陌生人。
        # 需要模拟"换一台设备"时传 carry_visitor=False。
        app = kwargs.get("app", args[0] if args else None)
        root = getattr(getattr(app, "state", None), "store", None)
        root = getattr(root, "root", None)
        if root is not None and carry:
            for other in reversed(clients):
                if getattr(other.app.state, "store", None).root != root:
                    continue
                token = other.cookies.get("la_visitor")
                if token:
                    self.cookies.update(other.cookies)
                break
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
