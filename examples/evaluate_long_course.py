"""Opt-in local ASR and real-process recovery. No chat API or teaching-quality verdict."""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.jobs import Jobs
from learning_agent.store import Store
from learning_agent.tools import MaterialTools


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def upload(client, space, path, interrupt=False):
    session = client.post(
        f"/api/spaces/{space}/uploads",
        json={
            "filename": path.name,
            "size": path.stat().st_size,
        },
    ).json()
    identifier = session["id"]
    with path.open("rb") as file:
        while block := file.read(session["chunk_size"]):
            response = client.patch(
                f"/api/uploads/{identifier}",
                content=block,
                headers={"Upload-Offset": str(session["received"])},
            )
            response.raise_for_status()
            session = response.json()
            if interrupt:
                # Query the authoritative offset as a reconnecting client would.
                # Actual browser reload/resume is validated separately.
                session = client.get(f"/api/uploads/{identifier}").json()
                file.seek(session["received"])
                interrupt = False
    response = client.post(f"/api/uploads/{identifier}/complete")
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("english", type=Path)
    parser.add_argument("--chinese", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "report.json"
    report = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"assets": {}}
    settings = Settings(
        _env_file=None,
        data_root=args.output / "workspace",
        asr_provider="local",
        asr_model=args.model,
        embedding_provider="disabled",
    )
    store, jobs = Store(settings.data_root), Jobs(Store(settings.data_root))
    with TestClient(create_app(settings)) as client:
        if "space_id" not in report:
            report["space_id"] = client.post(
                "/api/spaces", json={"content": "完整长课程验收"}
            ).json()["id"]
        space = report["space_id"]
        for key, path in [("english", args.english), ("chinese", args.chinese)]:
            if path is None or key in report["assets"]:
                continue
            start = time.monotonic()
            item = upload(client, space, path, interrupt=True)
            report["assets"][key] = {
                "id": item["id"],
                "file_bytes": path.stat().st_size,
                "duration": item["duration"],
                "import_s": time.monotonic() - start,
            }
            save(manifest_path, report)
        if args.chinese and args.captions:
            identifier = report["assets"]["chinese"]["id"]
            response = client.post(
                f"/api/spaces/{space}/materials/{identifier}/captions",
                files={"file": (args.captions.name, args.captions.read_bytes())},
            )
            response.raise_for_status()
        if not args.import_only and "job_id" not in report:
            mid = report["assets"]["english"]["id"]
            job = client.post(f"/api/spaces/{space}/materials/{mid}/transcribe").json()
            report["job_id"] = job["id"]
            save(manifest_path, report)
        report["api_driver_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        if args.import_only:
            save(manifest_path, report)
            return

        environment = dict(
            os.environ,
            LA_DATA_ROOT=str(settings.data_root.resolve()),
            LA_ASR_PROVIDER="local",
            LA_ASR_MODEL=args.model,
            LA_ASR_BASE_URL=settings.asr_base_url,
            LA_ASR_LOCAL_PART_SECONDS=str(settings.asr_local_part_seconds),
            LA_ASR_CONTEXT_SECONDS=str(settings.asr_context_seconds),
            LA_EMBEDDING_PROVIDER="disabled",
        )
        started = time.monotonic()
        peak = 0
        with (args.output / "worker.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "learning_agent.worker", "--drain"],
                env=environment,
                stdout=log,
                stderr=log,
            )
            try:
                while True:
                    status = jobs.get(report["job_id"])
                    if process.poll() is None:
                        sample = subprocess.run(
                            ["ps", "-o", "rss=", "-p", str(process.pid)],
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        value = sample.stdout.strip()
                        peak = max(peak, int(value or 0) * 1024)
                    if (
                        status["completed"] >= 2
                        and not report.get("killed_after_parts")
                        and status["state"] == "running"
                    ):
                        report["killed_after_parts"] = status["completed"]
                        process.kill()
                        process.wait(timeout=10)
                        # An inherited media-child lock may briefly outlive SIGKILL.
                        time.sleep(1)
                        process = subprocess.Popen(
                            [sys.executable, "-m", "learning_agent.worker", "--drain"],
                            env=environment,
                            stdout=log,
                            stderr=log,
                        )
                        save(manifest_path, report)
                    print(
                        json.dumps(
                            {
                                "state": status["state"],
                                "parts": status["completed"],
                                "total": status["total"],
                                "seconds": status["processed_seconds"],
                            }
                        ),
                        flush=True,
                    )
                    if status["state"] in {"succeeded", "failed", "paused", "cancelled"}:
                        break
                    if process.poll() is not None:
                        raise RuntimeError("处理进程提前退出，查看本机worker.log后重跑相同命令")
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)
        report["processing_s"] = time.monotonic() - started
        report["worker_sampled_peak_rss_bytes"] = peak
        report["job"] = jobs.get(report["job_id"])
        with store.connect() as db:
            report["part_attempts"] = {
                r[0]: json.loads(r[1]).get("created_attempt")
                for r in db.execute(
                    "SELECT part,result FROM job_parts WHERE job_id=? AND part LIKE 'asr:%'",
                    (report["job_id"],),
                )
            }
        report["late_observations"] = {}
        for key, asset in report["assets"].items():
            info = store.media(space, asset["id"])
            asset["media_state"], asset["transcript_state"] = (
                info["state"],
                info["transcript_state"],
            )
            if info["state"] == "ready":
                response = client.get(
                    f"/api/spaces/{space}/materials/{asset['id']}/playback",
                    headers={"Range": "bytes=0-127"},
                )
                asset["range_status"] = response.status_code
            report["late_observations"][key] = MaterialTools(store, space, False).execute(
                "inspect_media",
                {
                    "material_id": asset["id"],
                    "start_seconds": min(3700, max(0, info["duration"] - 30)),
                    "end_seconds": min(3730, info["duration"]),
                    "mode": "transcript",
                },
            )
        report["review"] = "工程检查；实际ASR边界及教学内容需另行人工复核"
        save(manifest_path, report)
        print(
            json.dumps({"report": str(manifest_path), "state": report["job"]["state"]}), flush=True
        )
        if report["job"]["state"] != "succeeded":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
