import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_learning import Scripted
from test_media import clip  # noqa: F401 -- pytest fixture

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.store import Store
from learning_agent.uploads import Uploads


def setup(tmp_path):
    settings = Settings(_env_file=None, data_root=tmp_path)
    client = TestClient(create_app(settings, Scripted([])))
    space = client.post("/api/spaces", json={"content": "course"}).json()["id"]
    return settings, client, space


def test_resume_drops_unacknowledged_tail_and_complete_is_idempotent(tmp_path, clip):  # noqa: F811
    settings, client, space = setup(tmp_path)
    item = client.post(
        f"/api/spaces/{space}/uploads", json={"filename": "full.mp4", "size": len(clip)}
    ).json()
    base = "/api/uploads/" + item["id"]
    assert (
        client.patch(base, content=clip[:100], headers={"Upload-Offset": "0"}).json()["received"]
        == 100
    )
    path = tmp_path / "uploads" / item["id"] / "upload.part"
    with path.open("ab") as file:
        file.write(b"unconfirmed tail")
    client = TestClient(create_app(settings, Scripted([])))
    assert client.get(base).json()["received"] == 100
    assert client.patch(base, content=b"wrong", headers={"Upload-Offset": "0"}).status_code == 409
    assert client.post(base + "/complete").status_code == 409
    assert (
        client.patch(base, content=clip[100:], headers={"Upload-Offset": "100"}).status_code == 200
    )
    assert path.read_bytes() == clip
    first = client.post(base + "/complete")
    assert first.status_code == 202
    second = client.post(base + "/complete").json()
    assert second["id"] == first.json()["id"]
    assert second["job_id"] == first.json()["job_id"]
    assert len(client.get(f"/api/spaces/{space}/materials").json()) == 1
    assert client.delete(base).status_code == 409


def test_move_before_registration_crash_can_complete_again(tmp_path, clip):  # noqa: F811
    _, client, space = setup(tmp_path)
    item = client.post(
        f"/api/spaces/{space}/uploads", json={"filename": "full.mp4", "size": len(clip)}
    ).json()
    base = "/api/uploads/" + item["id"]
    client.patch(base, content=clip, headers={"Upload-Offset": "0"}).raise_for_status()
    directory = tmp_path / "materials" / item["material_id"]
    directory.mkdir()
    (tmp_path / "uploads" / item["id"] / "upload.part").replace(directory / "original.mp4")
    assert client.post(base + "/complete").status_code == 202


def test_upload_limits_bad_media_cancel_and_missing_header(tmp_path):
    _, client, space = setup(tmp_path)
    url = f"/api/spaces/{space}/uploads"
    assert client.post(url, json={"filename": "x.mp4", "size": 2 * 1024**3 + 1}).status_code == 413
    assert client.post(url, json={"filename": "x.txt", "size": 4}).status_code == 422
    item = client.post(url, json={"filename": "x.mp4", "size": 4}).json()
    base = "/api/uploads/" + item["id"]
    assert client.patch(base, content=b"data").status_code == 422
    assert (
        client.patch(base, content=b"too much", headers={"Upload-Offset": "0"}).status_code == 413
    )
    assert client.get(base).json()["received"] == 0
    assert client.patch(base, content=b"data", headers={"Upload-Offset": "0"}).status_code == 200
    assert client.post(base + "/complete").status_code == 422
    assert client.delete(base).json()["state"] == "cancelled"
    assert not (tmp_path / "materials" / item["material_id"] / "original.mp4").exists()
    assert client.get("/api/uploads/missing").status_code == 404


def test_same_upload_is_serialized_without_holding_a_database_transaction(tmp_path):
    settings, _, space = setup(tmp_path)
    store = Store(tmp_path)
    uploads = Uploads(store, Jobs(store), settings)
    item = uploads.create(space, "x.mp4", 10)

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()

        async def slow():
            yield b"first"
            started.set()
            await release.wait()

        first = asyncio.create_task(uploads.append(item["id"], 0, slow()))
        await started.wait()
        store.add_material(space, "notes.txt", "Database writes stay available")
        with pytest.raises(HTTPException) as error:
            await uploads.append(item["id"], 0, slow())
        assert error.value.status_code == 409
        release.set()
        assert (await first)["received"] == 5

    asyncio.run(exercise())


def test_disk_full_preserves_acknowledged_bytes(tmp_path, monkeypatch):
    from learning_agent import uploads as module

    settings, _, space = setup(tmp_path)
    store = Store(tmp_path)
    uploads = Uploads(store, Jobs(store), settings)
    item = uploads.create(space, "x.mp4", 10)
    real_call = module.disk_call

    async def fail(function, *args):
        if function.__name__ == "write":
            raise OSError("disk full")
        return await real_call(function, *args)

    async def stream():
        yield b"first"

    monkeypatch.setattr(module, "disk_call", fail)
    with pytest.raises(HTTPException) as error:
        asyncio.run(uploads.append(item["id"], 0, stream()))
    assert error.value.status_code == 507
    assert uploads.get(item["id"])["received"] == 0
    monkeypatch.setattr(module, "disk_call", real_call)
    assert asyncio.run(uploads.append(item["id"], 0, stream()))["received"] == 5
