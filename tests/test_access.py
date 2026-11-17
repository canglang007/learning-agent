"""访问口令：没配时完全不变，配了之后页面和接口都要先登录。"""

from client_helpers import TestClient, close_for

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.store import Store


def settings(root, code=""):
    item = Settings(
        _env_file=None,
        data_root=root,
        retrieval_mode="bm25",
        embedding_provider="disabled",
        access_token=code,
    )
    Store(root)
    return item


def test_no_token_keeps_everything_open(tmp_path):
    root = tmp_path / "open"
    client = TestClient(create_app(settings(root)))
    try:
        assert client.get("/").status_code == 200
        assert client.get("/api/spaces").status_code == 200
        # 没有口令就没有登录页，避免给人“已经受保护”的错觉
        assert client.get("/login").status_code == 404
    finally:
        close_for(root)


def test_token_redirects_pages_and_rejects_api(tmp_path):
    root = tmp_path / "guarded"
    client = TestClient(create_app(settings(root, "open-sesame")))
    try:
        page = client.get("/", follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"].startswith("/login")
        api = client.get("/api/spaces")
        assert api.status_code == 401
        assert "口令" in api.json()["detail"]
    finally:
        close_for(root)


def test_login_sets_cookie_and_unlocks_api(tmp_path):
    root = tmp_path / "login"
    client = TestClient(create_app(settings(root, "open-sesame")))
    try:
        assert client.post("/login", data={"code": "wrong"}).status_code == 401
        assert client.get("/api/spaces").status_code == 401
        done = client.post(
            "/login", data={"code": "open-sesame", "next": "/"}, follow_redirects=False
        )
        assert done.status_code == 303
        assert client.cookies.get("la_access")
        assert client.get("/api/spaces").status_code == 200
        assert client.get("/").status_code == 200
    finally:
        close_for(root)


def test_health_and_static_open_without_login(tmp_path):
    root = tmp_path / "assets"
    client = TestClient(create_app(settings(root, "open-sesame")))
    try:
        assert client.get("/healthz").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/status").status_code == 401
    finally:
        close_for(root)


def test_logout_drops_the_session(tmp_path):
    root = tmp_path / "logout"
    client = TestClient(create_app(settings(root, "open-sesame")))
    try:
        client.post("/login", data={"code": "open-sesame"}, follow_redirects=False)
        assert client.get("/api/spaces").status_code == 200
        assert client.post("/logout", follow_redirects=False).status_code == 303
        assert client.get("/api/spaces").status_code == 401
    finally:
        close_for(root)


def test_repeated_failures_are_throttled(tmp_path):
    root = tmp_path / "brute"
    client = TestClient(create_app(settings(root, "open-sesame")))
    try:
        codes = [client.post("/login", data={"code": "nope"}).status_code for _ in range(9)]
        assert 429 in codes
        assert client.post("/login", data={"code": "open-sesame"}).status_code == 429
    finally:
        close_for(root)


def test_next_target_cannot_leave_the_site(tmp_path):
    root = tmp_path / "redirect"
    client = TestClient(create_app(settings(root, "open-sesame")))
    try:
        done = client.post(
            "/login",
            data={"code": "open-sesame", "next": "https://example.com/x"},
            follow_redirects=False,
        )
        assert done.status_code == 303
        assert done.headers["location"] == "/"
    finally:
        close_for(root)
