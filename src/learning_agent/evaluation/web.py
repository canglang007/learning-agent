import re
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from learning_agent.evaluation.ledger import Ledger
from learning_agent.evaluation.report import report


def install(app, settings):
    root = (settings.evaluation_root or settings.data_root / "evaluations").resolve()

    def experiment(eid):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", eid):
            raise HTTPException(404, "评测不存在")
        path = (root / eid).resolve()
        if not path.is_relative_to(root) or not (path / "experiment.db").is_file():
            raise HTTPException(404, "评测不存在")
        return path

    @app.get("/evaluation", response_class=HTMLResponse)
    async def page():
        return Path(__file__).with_name("view.html").read_text()

    @app.get("/api/evaluations")
    async def experiments():
        result = []
        for path in sorted(root.glob("*/experiment.db")):
            if not path.resolve().is_relative_to(root):
                continue
            ledger = Ledger(path, readonly=True)
            manifest = ledger.get("manifest")
            if manifest:
                result.append({"id": path.parent.name, **manifest})
        return result

    @app.get("/api/evaluations/{eid}")
    async def overview(eid: str):
        return report(experiment(eid), readonly=True)

    @app.get("/api/evaluations/{eid}/trials/{tid}")
    async def trial(eid: str, tid: str):
        rows = [
            r
            for r in report(experiment(eid), include_content=True, readonly=True)["rows"]
            if r["trial_id"] == tid
        ]
        if not rows:
            raise HTTPException(404, "Trial不存在")
        return {"rows": rows}

    @app.get("/api/evaluations/{eid}/sources/{tid}/{alias}")
    async def source(eid: str, tid: str, alias: str):
        path = experiment(eid)
        ledger = Ledger(path / "experiment.db", readonly=True)
        if not ledger.get("trial:" + tid):
            raise HTTPException(404, "Trial不存在")
        asset = ledger.get("corpus", {}).get("assets", {}).get(alias)
        if not asset:
            raise HTTPException(404, "资料不存在")
        base = (path / "trials" / tid / "workspace" / "materials").resolve()
        candidates = [
            base / (asset["id"] + ".pdf"),
            # 直接可播的资料没有 preview.mp4，原文件就是 original.<后缀>（可能是 mkv/webm）。
            *sorted((base / asset["id"]).glob("original.*")),
            base / asset["id"] / "preview.mp4",
            base / asset["id"] / "preview.m4a",
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve().is_relative_to(base):
                return FileResponse(candidate)
        raise HTTPException(404, "没有可播放或查看的原始资料")

    @app.get("/api/evaluations/{eid}/images/{tid}/{operation_id}/{index}")
    async def image(eid: str, tid: str, operation_id: str, index: int):
        path = experiment(eid)
        trial = Ledger(path / "experiment.db", readonly=True).get("trial:" + tid)
        if not trial or index < 0:
            raise HTTPException(404, "图像不存在")
        for item in trial["turns"].values():
            for op in item.get("trace", {}).get("operations", []):
                images = (op.get("input_ref") or {}).get("images", [])
                if op["id"] == operation_id and index < len(images):
                    base = (path / "trials" / tid / "workspace").resolve()
                    target = (base / images[index]["asset_path"]).resolve()
                    if target.is_relative_to(base) and target.suffix == ".png" and target.is_file():
                        return FileResponse(target, media_type="image/png")
        raise HTTPException(404, "图像不存在")
