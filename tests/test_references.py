from learning_agent.references import resolve_references
from learning_agent.store import Store


def event(observation):
    return {"type": "tool", "name": "inspect_media", "observation": observation}


def media(material="video-1", start=0, end=60):
    return {
        "name": "course.mp4",
        "material_id": material,
        "duration": 120,
        "start_seconds": start,
        "end_seconds": end,
    }


def test_citations_resolve_only_observed_ranges_and_pages():
    events = [
        event(media()),
        event(media(start=60, end=100)),
        event(
            {"name": "slides.pdf", "material_id": "pdf-1", "page": 14, "page_url": "/actual/page14"}
        ),
    ]
    refs = resolve_references(
        "[course.mp4:00:25-01:15] [course.mp4:01:40-02:00] [course.mp4:00:60-01:20] [slides.pdf:p14] [slides.pdf:p15]",
        events,
    )
    assert refs[0]["position_verified"] and refs[0]["start_seconds"] == 25
    assert refs[0]["end_seconds"] == 75
    assert not refs[1]["position_verified"] and not refs[2]["position_verified"]
    assert refs[3]["url"] == "/actual/page14"
    assert refs[4]["reason"] == "page_not_observed"


def test_ambiguous_names_and_gaps_do_not_guess_links():
    text = "[course.mp4:00:10–00:20]"
    refs = resolve_references(text, [event(media()), event(media("different-video"))])
    assert refs[0]["reason"] == "ambiguous_or_unobserved_material"
    refs = resolve_references(
        "[course.mp4:00:10-01:20]", [event(media(end=40)), event(media(start=50, end=90))]
    )
    assert refs[0]["reason"] == "time_range_not_observed"
    assert not resolve_references(text, [event({"error": "failed", **media()})])[0][
        "position_verified"
    ]


def test_search_excerpt_can_support_position_without_claiming_page_image_read():
    refs = resolve_references(
        "[slides.pdf:p15]",
        [
            event(
                {
                    "hits": [
                        {
                            "id": "pdf-1",
                            "name": "slides.pdf",
                            "page": 15,
                            "excerpt": "Actual returned text",
                        }
                    ]
                }
            )
        ],
        "space-1",
    )
    assert refs[0]["position_verified"]
    assert refs[0]["evidence_kind"] == "search_excerpt"
    assert refs[0]["url"] == "/api/spaces/space-1/materials/pdf-1/pages/15/image"
    refs = resolve_references(
        "[course.mp4:00:02-00:04]",
        [
            event(
                {
                    "hits": [
                        {
                            "id": "video-1",
                            "name": "course.mp4",
                            "start_seconds": 1.5,
                            "end_seconds": 4.5,
                            "excerpt": "Actual caption",
                        }
                    ]
                }
            )
        ],
        "space-1",
    )
    assert refs[0]["position_verified"] and refs[0]["start_seconds"] == 2


def test_followups_can_reuse_prior_evidence_but_not_other_conversations_or_future(tmp_path):
    store = Store(tmp_path)
    space = store.space("study")
    conv = store.conversation(space["id"])
    first = store.start_run(conv["id"], "first")
    store.save_events(first, [event(media(end=30))])
    store.finish_run(first, "completed", "[course.mp4:00:10-00:20]", [])
    followup = store.start_run(conv["id"], "followup")
    store.finish_run(followup, "completed", "[course.mp4:00:15-00:25]", [])
    other = store.conversation(space["id"])
    other_run = store.start_run(other["id"], "other")
    store.save_events(other_run, [event(media("different-video", end=100))])
    store.finish_run(other_run, "completed", "other", [])
    future = store.start_run(conv["id"], "future")
    store.save_events(future, [event(media(start=30, end=100))])
    store.finish_run(future, "completed", "future", [])
    answer, events = store.reference_context(followup)
    assert resolve_references(answer, events)[0]["position_verified"]
    assert len(events) == 1
    assert not resolve_references("[course.mp4:00:30-01:00]", events)[0]["position_verified"]
