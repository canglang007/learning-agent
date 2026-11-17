"""账号：邀请码注册、登录态持久化、会话与学习记忆按账号隔离。

口令只回答"能不能进来"，账号还要回答"这是谁"——所以这里既测门禁，也测归属。
"""

from client_helpers import TestClient
from fastapi.testclient import TestClient as RawClient

from learning_agent.accounts import check_password
from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.store import Store

PASSWORD = "correct-horse"


def settings(root):
    return Settings(
        _env_file=None,
        data_root=root,
        auth="accounts",
        retrieval_mode="bm25",
        embedding_provider="disabled",
    )


def invite(root, uses=1, note="test"):
    return Store(root).create_invite(note=note, max_uses=uses)


class Server:
    """一个应用 + 多个浏览器：只有第一个 client 启动 lifespan，避免重复起调度器。"""

    def __init__(self, root):
        self.root = root
        self.app = create_app(settings(root))
        self.started = []
        self.spare = []

    def browser(self):
        if not self.started:
            client = TestClient(self.app)
            self.started.append(client)
        else:
            client = RawClient(self.app)
            self.spare.append(client)
        return client

    def close(self):
        for client in self.started + self.spare:
            client.close()


def register(client, code, username, password=PASSWORD, **extra):
    data = {
        "invite": code,
        "username": username,
        "password": password,
        "confirm": password,
        "next": "/",
        **extra,
    }
    return client.post("/register", data=data, follow_redirects=False)


def login(client, username, password=PASSWORD, **extra):
    data = {"username": username, "password": password, "next": "/", **extra}
    return client.post("/login", data=data, follow_redirects=False)


def test_page_and_api_both_require_login(tmp_path):
    server = Server(tmp_path)
    try:
        guest = server.browser()
        page = guest.get("/", follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"].startswith("/login")
        assert guest.get("/api/spaces").status_code == 401
        assert guest.get("/login").status_code == 200
        assert guest.get("/api/me").json()["authenticated"] is False
    finally:
        server.close()


def test_register_with_invite_opens_the_app(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        alice = server.browser()
        done = register(alice, code, "alice")
        assert done.status_code == 303
        assert done.headers["location"] == "/"
        me = alice.get("/api/me").json()
        assert me["authenticated"] is True
        assert me["username"] == "alice"
        assert alice.get("/").status_code == 200
        assert alice.get("/api/spaces").status_code == 200
    finally:
        server.close()


def test_password_is_never_stored_in_clear(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        register(server.browser(), code, "alice")
        stored = Store(tmp_path).user_by_name("alice")["password"]
        assert stored.startswith("scrypt$")
        assert PASSWORD not in stored
        assert check_password(stored, PASSWORD)
        assert not check_password(stored, "wrong-password")
    finally:
        server.close()


def test_bad_invite_creates_nothing(tmp_path):
    invite(tmp_path, note="留给别人")
    server = Server(tmp_path)
    try:
        bob = server.browser()
        assert register(bob, "not-a-real-code", "bob").status_code == 422
        assert Store(tmp_path).has_users() is False
        assert Store(tmp_path).invites()[0]["uses"] == 0
        assert bob.get("/api/spaces").status_code == 401
    finally:
        server.close()


def test_duplicate_name_does_not_burn_the_invite(tmp_path):
    code = invite(tmp_path, uses=2)
    server = Server(tmp_path)
    try:
        register(server.browser(), code, "alice")
        second = register(server.browser(), code, "ALICE")  # 大小写也算重名
        assert second.status_code == 422
        assert "已被占用" in second.text
        assert Store(tmp_path).invites()[0]["uses"] == 1
    finally:
        server.close()


def test_short_password_is_rejected(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        bob = server.browser()
        assert register(bob, code, "bob", password="short").status_code == 422
        assert Store(tmp_path).has_users() is False
    finally:
        server.close()


def test_wrong_password_is_rate_limited(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        register(server.browser(), code, "alice")
        guest = server.browser()
        for _ in range(8):
            assert login(guest, "alice", "wrong-password").status_code == 401
        assert login(guest, "alice", "wrong-password").status_code == 429
        # 锁定只针对失败方，不影响正确密码走别的入口——这里先确认锁仍然生效
        assert login(guest, "alice").status_code == 429
    finally:
        server.close()


def test_unknown_user_looks_the_same_as_wrong_password(tmp_path):
    server = Server(tmp_path)
    try:
        guest = server.browser()
        assert login(guest, "nobody").status_code == 401
        assert "用户名或密码不正确" in login(guest, "nobody").text
    finally:
        server.close()


def test_session_survives_restart_and_keeps_conversations(tmp_path):
    code = invite(tmp_path)
    first = Server(tmp_path)
    alice = first.browser()
    register(alice, code, "alice")
    sid = alice.post("/api/spaces", json={"content": "共用资料"}).json()["id"]
    conversation = alice.post(f"/api/spaces/{sid}/conversations").json()["id"]
    cookie = alice.cookies.get("la_session")
    first.close()

    second = Server(tmp_path)
    try:
        reopened = second.browser()
        reopened.cookies.set("la_session", cookie)
        assert reopened.get("/api/me").json()["username"] == "alice"
        listed = reopened.get(f"/api/spaces/{sid}/conversations").json()
        assert [item["id"] for item in listed] == [conversation]
    finally:
        second.close()


def test_two_accounts_are_isolated(tmp_path):
    code = invite(tmp_path, uses=2)
    server = Server(tmp_path)
    try:
        alice, bob = server.browser(), server.browser()
        register(alice, code, "alice")
        register(bob, code, "bob")
        sid = alice.post("/api/spaces", json={"content": "共用资料"}).json()["id"]
        alice_conversation = alice.post(f"/api/spaces/{sid}/conversations").json()["id"]
        bob_conversation = bob.post(f"/api/spaces/{sid}/conversations").json()["id"]

        assert [c["id"] for c in alice.get(f"/api/spaces/{sid}/conversations").json()] == [
            alice_conversation
        ]
        assert [c["id"] for c in bob.get(f"/api/spaces/{sid}/conversations").json()] == [
            bob_conversation
        ]
        assert bob.get(f"/api/conversations/{alice_conversation}/messages").status_code == 404
        assert bob.delete(f"/api/conversations/{alice_conversation}").status_code == 404
        # 资料仍然共用
        assert sid in [space["id"] for space in bob.get("/api/spaces").json()]
    finally:
        server.close()


def test_topics_and_memories_belong_to_the_signed_in_account(tmp_path):
    code = invite(tmp_path, uses=2)
    server = Server(tmp_path)
    try:
        alice, bob = server.browser(), server.browser()
        register(alice, code, "alice")
        register(bob, code, "bob")
        sid = alice.post("/api/spaces", json={"content": "共用资料"}).json()["id"]
        topic = alice.post(
            f"/api/spaces/{sid}/learning-topics", json={"title": "梯度", "material_ids": []}
        ).json()
        memory = alice.post(
            f"/api/spaces/{sid}/memories",
            json={
                "topic_id": topic["id"],
                "kind": "preference",
                "text": "先给数值例子再讲公式",
                "material_ids": [],
            },
        ).json()

        assert bob.get(f"/api/spaces/{sid}/learning-topics").json() == []
        assert bob.get(
            f"/api/spaces/{sid}/memories", params={"topic_id": topic["id"]}
        ).status_code == 404
        assert bob.post(
            f"/api/spaces/{sid}/memories",
            json={"topic_id": topic["id"], "kind": "goal", "text": "偷看别人的", "material_ids": []},
        ).status_code == 404
        assert bob.patch(
            f"/api/spaces/{sid}/memories/{memory['id']}",
            json={"revision": memory["revision"], "text": "改掉别人的"},
        ).status_code == 404

        mine = alice.get(f"/api/spaces/{sid}/memories", params={"topic_id": topic["id"]}).json()
        assert [item["text"] for item in mine] == ["先给数值例子再讲公式"]
    finally:
        server.close()


def test_new_password_ends_old_sessions(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        alice = server.browser()
        register(alice, code, "alice")
        user = Store(tmp_path).user_by_name("alice")
        Store(tmp_path).set_password(user["id"], "brand-new-secret")

        assert alice.get("/api/spaces").status_code == 401
        assert login(alice, "alice", "brand-new-secret").status_code == 303
        assert alice.get("/api/spaces").status_code == 200
    finally:
        server.close()


def test_disable_blocks_login_and_kills_live_session(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        alice = server.browser()
        register(alice, code, "alice")
        user = Store(tmp_path).user_by_name("alice")
        Store(tmp_path).set_disabled(user["id"], True)

        assert alice.get("/api/spaces").status_code == 401
        assert login(alice, "alice").status_code == 401
    finally:
        server.close()


def test_logout_drops_the_session(tmp_path):
    code = invite(tmp_path)
    server = Server(tmp_path)
    try:
        alice = server.browser()
        register(alice, code, "alice")
        assert alice.post("/logout", follow_redirects=False).status_code == 303
        assert alice.get("/api/spaces").status_code == 401
        assert alice.get("/login").status_code == 200
    finally:
        server.close()


def test_signup_closes_once_invites_run_out(tmp_path):
    code = invite(tmp_path, uses=1)
    server = Server(tmp_path)
    try:
        register(server.browser(), code, "alice")
        bob = server.browser()
        page = bob.get("/register")
        assert "注册暂未开放" in page.text
        assert register(bob, code, "bob").status_code == 403
    finally:
        server.close()


def test_anonymous_mode_still_works_without_accounts(tmp_path):
    """没账号、没邀请码、没口令时，保持原来的本地开发行为。"""
    client = TestClient(create_app(Settings(_env_file=None, data_root=tmp_path)))
    try:
        assert client.get("/").status_code == 200
        assert client.get("/api/me").json()["mode"] == "off"
        assert client.get("/login").status_code == 404
    finally:
        client.close()
