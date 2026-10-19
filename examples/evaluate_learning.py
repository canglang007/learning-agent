"""Run a real-material, multi-turn learning suite. No automatic teaching-quality verdict."""

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.media import command, probe
from learning_agent.store import Store
from learning_agent.worker import Worker


def save(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def evidence_summary(events, answer, assets, prior_events=()):
    """Mechanical evidence checks are separate from correctness/teaching review."""
    observations = [e["observation"] for e in [*prior_events, *events] if e["type"] == "tool"]
    pages, frames, ranges, errors = [], [], [], []
    for event in events:
        if event["type"] != "tool":
            continue
        o = event["observation"]
        if o.get("error"):
            errors.append({"tool": event["name"], "error": o["error"]})
        elif o.get("image"):
            pages.append({"material_id": o["material_id"], "page": o["page"]})
        if "start_seconds" in o and "segments" in o:
            ranges.append(
                {
                    "material_id": o["material_id"],
                    "start": o["start_seconds"],
                    "end": o["end_seconds"],
                    "caption_coverage_ratio": o["caption_coverage_ratio"],
                }
            )
            frames.extend(
                {"material_id": o["material_id"], "seconds": image["seconds"]}
                for image in o["images"]
            )
    links = []
    for match in re.finditer(r"/#media=([^\s)\]<>]+)", answer):
        query = parse_qs("media=" + match[1])
        material_id = query.get("media", [""])[0]
        asset = next((a for a in assets.values() if a["id"] == material_id), None)
        try:
            a, b = float(query["start"][0]), float(query["end"][0])
            valid = asset is not None and 0 <= a < b <= asset.get("duration", 0)
        except (ValueError, KeyError):
            a = b = None
            valid = False
        observed = valid and any(
            o.get("material_id") == material_id
            and o.get("start_seconds", float("inf")) <= a
            and o.get("end_seconds", -1) >= b
            for o in observations
        )
        offset = asset.get("source_start_seconds", 0) if asset else 0
        links.append(
            {
                "url": match[0],
                "within_material": valid,
                "within_observed_interval": observed,
                "original_start_seconds": a + offset if valid else None,
                "original_end_seconds": b + offset if valid else None,
            }
        )
    return {
        "viewed_pdf_pages": pages,
        "viewed_frames": frames,
        "media_intervals": ranges,
        "tool_errors": errors,
        "playback_links": links,
        "note": "仅核对实际观察与位置，不代表讲解正确、回听内容相关或用户已经学会。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="配合--case追加一次新对话，保留原结果")
    options = parser.parse_args()
    if options.rerun and not options.cases:
        parser.error("--rerun需要用--case指定复测的任务组")
    suite = json.loads(options.suite.read_text(encoding="utf-8"))
    settings = Settings()
    if not options.prepare_only and not settings.api_key.get_secret_value():
        raise SystemExit("请先配置 LA_API_KEY；不输出密钥。")
    # Resolve every input before starting model calls or copying large materials.
    for item in suite["assets"].values():
        if "content" in item and not Path(item["path"]).exists():
            target = Path(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["content"], encoding="utf-8")
        if not Path(item["path"]).is_file():
            raise SystemExit("缺少资料：" + item["path"])
        if item.get("transcribe") and "source_end_seconds" in item:
            expected = item["source_end_seconds"] - item["source_start_seconds"]
            actual = probe(Path(item["path"]))["duration"]
            streams = json.loads(
                command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", item["path"]])
            )["streams"]
            # Remote range downloads can exit successfully with a truncated audio track.
            tracks = [
                float(s.get("duration", actual))
                for s in streams
                if s["codec_type"] in {"audio", "video"}
            ]
            if abs(actual - expected) > 0.5 or any(abs(t - expected) > 0.5 for t in tracks):
                raise SystemExit("截取素材/音轨时长不完整，请先核对并重新截取：" + item["path"])
    root = options.output or settings.data_root / "evaluations" / datetime.now(UTC).strftime(
        "learning-%Y%m%dT%H%M%S%fZ"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.json"
    report = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "suite": str(options.suite),
            "model": settings.model,
            "protocol": settings.protocol,
            "assets": {},
            "cases": [],
            "review_method": "manual_rubric_pending; no model-judge pass rate",
        }
    )
    settings = settings.model_copy(update={"data_root": root / "workspace"})
    store = Store(settings.data_root)
    with TestClient(create_app(settings)) as client, Worker(settings) as worker:
        if "space_id" not in report:
            report["space_id"] = client.post("/api/spaces", json={"content": suite["name"]}).json()[
                "id"
            ]
        space = report["space_id"]
        for key, spec in suite["assets"].items():
            if key in report["assets"]:
                continue
            source = Path(spec["path"])
            start = time.monotonic()
            response = client.post(
                f"/api/spaces/{space}/materials", files={"file": (source.name, source.read_bytes())}
            )
            response.raise_for_status()
            material_id = response.json()["id"]
            worker.drain()
            asset = {**spec, "id": material_id, "ingestion_s": round(time.monotonic() - start, 2)}
            if spec.get("transcribe"):
                start = time.monotonic()
                response = client.post(f"/api/spaces/{space}/materials/{material_id}/transcribe")
                response.raise_for_status()
                worker.drain()
                info = client.get(f"/api/spaces/{space}/materials/{material_id}/media").json()
                if info["state"] != "ready" or info["transcript_state"] != "ready":
                    raise SystemExit(
                        "媒体/转写未就绪：" + str(info.get("transcript_error") or info.get("error"))
                    )
                asset.update(
                    {
                        "duration": info["duration"],
                        "transcription_s": round(time.monotonic() - start, 2),
                        "transcript_source": info["transcript_source"],
                        "alignment_notes": info.get("alignment_notes", []),
                    }
                )
                save(root / f"{key}-transcript.json", store.segments(space, material_id))
            report["assets"][key] = asset
            save(path, report)
            print(
                json.dumps(
                    {
                        "prepared": key,
                        "duration": asset.get("duration"),
                        "asr_s": asset.get("transcription_s"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if options.prepare_only:
            print("资料准备完成：" + str(path), flush=True)
            return
        for case in suite["cases"]:
            if options.cases and case["id"] not in options.cases:
                continue
            attempts = sum(c["id"] == case["id"] for c in report["cases"])
            if attempts and not options.rerun:
                continue
            conv = client.post(f"/api/spaces/{space}/conversations").json()["id"]
            row = {"id": case["id"], "attempt": attempts + 1, "conversation_id": conv, "turns": []}
            report["cases"].append(row)
            prior_events = []
            for i, turn in enumerate(case["turns"]):
                payload = {"content": turn["question"]}
                if turn.get("focus"):
                    f = dict(turn["focus"])
                    f["material_id"] = report["assets"][f.pop("asset")]["id"]
                    payload["focus"] = f
                start = time.monotonic()
                response = client.post(f"/api/conversations/{conv}/messages", json=payload)
                response.raise_for_status()
                result = response.json()
                run = client.get(f"/api/runs/{result['run_id']}").json()
                events = run["events"]
                review = {
                    "accuracy_0_3": None,
                    "addresses_confusion_0_3": None,
                    "clarity_0_3": None,
                    "feedback_adaptation_0_3": None,
                    "serious_error": None,
                    "evidence_and_replay": None,
                    "reason": None,
                }
                row["turns"].append(
                    {
                        **result,
                        "question": turn["question"],
                        "focus": payload.get("focus"),
                        "latency_s": round(time.monotonic() - start, 2),
                        "rubric": turn["rubric"],
                        "events": events,
                        "resolved_references": run.get("references", []),
                        "evidence": evidence_summary(
                            events, result["answer"], report["assets"], prior_events
                        ),
                        "review": review,
                    }
                )
                prior_events.extend(events)
                save(path, report)
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "turn": i + 1,
                            "state": result["state"],
                            "seconds": row["turns"][-1]["latency_s"],
                            "tools": [e["name"] for e in events if e["type"] == "tool"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if result["state"] == "failed":
                    break
    print("待复核报告：" + str(path), flush=True)


if __name__ == "__main__":
    main()
