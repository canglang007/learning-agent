"""会话隔离：每个浏览器只看到自己建的会话，公共示例所有人可见。

口令负责"能不能进来"，访客标识负责"看到谁的东西"，两件事分开测。

一个应用实例（单进程、单调度器）+ 多个 client，才等价于"同一台服务器上多个浏览器"。
"""

from client_helpers import TestClient
from fastapi.testclient import TestClient as RawClient

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.store import Store


class Guests:
    """第一个 client 负责启动 lifespan，其余的只发请求，避免重复启动调度器。"""

    def __init__(self, tmp_path):
        self.app = create_app(Settings(_env_file=None, data_root=tmp_path))
        self.started = []
        self.spare = []

    def open(self):
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


def space(client, name="隔离测试"):
    return client.post("/api/spaces", json={"content": name}).json()["id"]


def listed(client, sid):
    return [item["id"] for item in client.get(f"/api/spaces/{sid}/conversations").json()]


def test_two_guests_do_not_see_each_others_conversations(tmp_path):
    guests = Guests(tmp_path)
    alice, bob = guests.open(), guests.open()
    sid = space(alice, "共享资料")
    alice_conversation = alice.post(f"/api/spaces/{sid}/conversations").json()["id"]
    bob_conversation = bob.post(f"/api/spaces/{sid}/conversations").json()["id"]

    assert listed(alice, sid) == [alice_conversation]
    assert listed(bob, sid) == [bob_conversation]
    assert alice_conversation != bob_conversation
    guests.close()


def test_other_guest_cannot_reach_conversation_by_id(tmp_path):
    guests = Guests(tmp_path)
    alice, bob = guests.open(), guests.open()
    sid = space(alice)
    conversation = alice.post(f"/api/spaces/{sid}/conversations").json()["id"]

    assert alice.get(f"/api/conversations/{conversation}/messages").status_code == 200
    # 越权按"不存在"处理，不透露 id 是否真实
    assert bob.get(f"/api/conversations/{conversation}/messages").status_code == 404
    assert bob.get(f"/api/conversations/{conversation}/runs").status_code == 404
    assert bob.get(f"/api/conversations/{conversation}/learning").status_code == 404
    assert bob.post(
        f"/api/conversations/{conversation}/messages", json={"content": "插一句"}
    ).status_code == 404
    guests.close()


def test_public_conversation_is_visible_to_everyone(tmp_path):
    guests = Guests(tmp_path)
    alice, bob = guests.open(), guests.open()
    sid = space(alice)
    public = Store(tmp_path).conversation(sid)["id"]  # owner 为空：示例 / 迁移前历史

    assert listed(alice, sid) == [public]
    assert listed(bob, sid) == [public]
    assert bob.get(f"/api/conversations/{public}/messages").status_code == 200
    guests.close()


def test_same_guest_keeps_own_list_after_restart(tmp_path):
    first = Guests(tmp_path)
    alice = first.open()
    sid = space(alice)
    conversation = alice.post(f"/api/spaces/{sid}/conversations").json()["id"]
    cookie = alice.cookies.get("la_visitor")
    first.close()

    # 服务重启后同一台浏览器重开：只有访客 cookie 延续
    second = Guests(tmp_path)
    reopened = second.open()
    reopened.cookies.set("la_visitor", cookie)
    assert listed(reopened, sid) == [conversation]

    stranger = second.open()  # 换一台设备
    assert listed(stranger, sid) == []
    second.close()
