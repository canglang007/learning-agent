"""Text changes must not leave the space on a stale or empty index.

Publishing subtitles, recognising scanned pages and editing a source all rewrite
a material's chunks. The indexer therefore queues its own rebuild; these tests
pin that wiring so the "search suddenly returns nothing until someone clicks a
button" failure cannot come back.
"""

import json

from client_helpers import TestClient, close_for
from test_learning import Scripted

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.store import Store


def setup(tmp_path, **kwargs):
    settings = Settings(_env_file=None, data_root=tmp_path / "workspace", **kwargs)
    store = Store(settings.data_root)
    space = store.space("course")["id"]
    return settings, store, space, Jobs(store)


def video(store, space, identifier="v"):
    """A ready media row; the media pipeline itself is covered by test_media."""
    with store.connect() as db:
        db.execute(
            "INSERT INTO materials(id,space_id,name,text,format) VALUES (?,?,'course.mp4','','video')",
            (identifier, space),
        )
        db.execute(
            "INSERT INTO media(material_id,metadata,state) VALUES (?,?,'ready')",
            (identifier, json.dumps({"duration": 120, "has_audio": True, "has_video": True})),
        )
    return identifier


def kinds(store, space):
    with store.connect() as db:
        return [row[0] for row in db.execute("SELECT kind FROM jobs WHERE space_id=?", (space,))]


def test_queue_index_is_skipped_without_an_embedding_provider(tmp_path):
    settings, store, space, jobs = setup(tmp_path, embedding_provider="disabled")
    assert jobs.queue_index(space, settings) is None
    assert kinds(store, space) == []


def test_queue_index_dedupes_while_the_version_is_unchanged(tmp_path):
    settings, store, space, jobs = setup(tmp_path, embedding_provider="ollama")
    store.add_material(space, "notes.txt", "attention")
    first = jobs.queue_index(space, settings)
    second = jobs.queue_index(space, settings)
    assert first["id"] == second["id"]
    assert kinds(store, space) == ["index"]


def test_uploaded_captions_queue_an_index_rebuild(tmp_path):
    settings, store, space, _jobs = setup(tmp_path, embedding_provider="ollama")
    close_for(tmp_path)
    client = TestClient(create_app(settings, Scripted([])))
    material = video(store, space)
    assert kinds(store, space) == []

    captions = b"WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nnormalisation and residual paths\n"
    response = client.post(
        f"/api/spaces/{space}/materials/{material}/captions",
        files={"file": ("course.vtt", captions)},
    )
    assert response.status_code == 200
    assert response.json()["segment_count"] == 1
    assert kinds(store, space) == ["index"]


def test_source_change_queues_an_index_rebuild(tmp_path):
    settings, store, space, _jobs = setup(tmp_path, embedding_provider="ollama")
    close_for(tmp_path)
    client = TestClient(create_app(settings, Scripted([])))
    material = video(store, space)

    response = client.put(
        f"/api/spaces/{space}/materials/{material}/source",
        json={"url": "https://example.org/lecture?t=1", "start_seconds": 353},
    )
    assert response.status_code == 200
    assert kinds(store, space) == ["index"]
