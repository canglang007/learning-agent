import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from learning_agent.learning import Explanation
from learning_agent.media import MediaError
from learning_agent.pdf import PdfError
from learning_agent.retrieval import Retriever
from learning_agent.store import Store


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ListArgs(Args):
    pass


class SearchArgs(Args):
    query: str = Field(min_length=1, max_length=300)
    material_id: str | None = None


class OutlineArgs(Args):
    material_id: str
    start: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=30)


class ContextArgs(Args):
    chunk_id: str
    before: int = Field(default=1, ge=0, le=2)
    after: int = Field(default=1, ge=0, le=2)


class ReadArgs(Args):
    material_id: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class PageArgs(Args):
    material_id: str
    page: int = Field(ge=1, le=200)


class InspectArgs(PageArgs):
    region: list[float] | None = Field(default=None, min_length=4, max_length=4)


class MediaArgs(Args):
    material_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    mode: Literal["transcript", "joint"]
    frame_count: int = Field(default=3, ge=1, le=3)


ARGUMENTS = {
    "list_materials": ListArgs,
    "search_materials": SearchArgs,
    "material_outline": OutlineArgs,
    "read_context": ContextArgs,
    "read_passage": ReadArgs,
    "read_page": PageArgs,
    "inspect_page": InspectArgs,
    "inspect_media": MediaArgs,
    "deliver_explanation": Explanation,
}
DESCRIPTIONS = {
    "list_materials": "查看当前空间的资料目录，包含格式与PDF物理页数。",
    "search_materials": "搜索资料段落：BM25与已就绪的多语言向量按RRF融合；不可用时明确降级。material_id=null搜索全空间，或指定资料缩小范围。结果包含chunk_id和原始页码/时间；无结果可改写查询或查目录。",
    "material_outline": "分页查看指定资料的导航。PDF按物理页首行、音视频按5分钟时间分组、文字按段落；不是语义章节识别。start为从0开始的目录偏移，limit最多30；next_start可继续翻页。",
    "read_context": "按搜索/目录返回的chunk_id读取原段及前后最多各2段。用于补充定义、条件、推导上下文，再决定是否读整页、查看图像或检查视频。字幕更新后旧chunk_id会失效，应重新搜索。",
    "read_passage": "读取资料原文与行号。范围最多50行；获取推导条件、定义或前后文。",
    "read_page": "读取PDF指定物理页的提取文字与行号。图表关系/公式排版不能仅凭文字判断，应inspect_page查看。扫描页可能没有提取文字。",
    "inspect_page": "实际查看PDF页面图像以理解图表、公式和排版。page是从1开始的物理页。region=null查看整页，或用左上角原点的归一化[left,top,right,bottom]裁切区域，值在0到1之间。",
    "inspect_media": "读取音视频指定时间段（最多120秒）。transcript读取有时间戳字幕；joint同时抽样最多3帧，受本轮共享图像预算限制。返回字幕来源、实际帧时间和回放链接。不是逐帧观看或直接听音频，缺字幕时不能编造讲话内容。",
    "deliver_explanation": "最终交付讲解与简短学习工作记录，一次完成、无需后续模型调用。answer为给用户的Markdown正文，其他字段记录主题、卡点及实际讲法，不代表掌握。必须独立于其他工具调用，先看完所需观察再交付。",
}


def definitions(vision=True):
    result = []
    for name, model in ARGUMENTS.items():
        if name == "inspect_page" and not vision:
            continue
        schema = model.model_json_schema()
        schema["required"] = list(schema.get("properties", {}))
        result.append(
            {"name": name, "description": DESCRIPTIONS[name], "parameters": schema, "strict": True}
        )
    return result


def terms(text):
    tokens = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.casefold())
    result = set()
    for token in tokens:
        if re.fullmatch(r"[\u3400-\u9fff]+", token) and len(token) > 2:
            result.update(token[i : i + 2] for i in range(len(token) - 1))
        else:
            result.add(token)
    return result


class MaterialTools:
    def __init__(self, store: Store, space_id: str, vision=True, retriever=None):
        self.store, self.space_id = store, space_id
        self.vision = vision
        self.retriever = retriever or Retriever(store)

    def execute(self, name, arguments, image_budget=3):
        if name not in ARGUMENTS:
            return {"error": "unknown_tool", "detail": "工具不存在"}
        try:
            args = ARGUMENTS[name].model_validate(arguments)
            if name == "deliver_explanation":
                return {
                    "error": "requires_conversation",
                    "detail": "学习工作记录由当前会话执行器处理",
                }
            if name == "material_outline":
                return self.retriever.index.outline(
                    self.space_id, args.material_id, args.start, args.limit
                )
            if name == "read_context":
                return self.retriever.index.context(
                    self.space_id, args.chunk_id, args.before, args.after
                )
            if (
                name == "search_materials"
                and self.retriever.settings.retrieval_mode != "lexical_baseline"
            ):
                return self.retriever.search(self.space_id, args.query, args.material_id)
            if name == "list_materials":
                items = self.store.materials(self.space_id)
                for item in items:
                    if item["format"] in {"audio", "video"}:
                        info = self.store.media(self.space_id, item["id"])
                        item.update(
                            {
                                key: info[key]
                                for key in [
                                    "duration",
                                    "has_audio",
                                    "has_video",
                                    "frame_index_seconds",
                                    "transcript_source",
                                ]
                            }
                        )
                return {"materials": items}
            if name == "inspect_media":
                info = self.store.media(self.space_id, args.material_id)
                start, end = args.start_seconds, args.end_seconds
                if not (
                    math.isfinite(start)
                    and math.isfinite(end)
                    and start < end <= info["duration"]
                    and end - start <= 120
                ):
                    return {
                        "error": "invalid_range",
                        "detail": "请选择媒体时长内顺序正确的片段，每次最多120秒",
                        "duration": info["duration"],
                    }
                segments = [
                    s
                    for s in self.store.segments(self.space_id, args.material_id)
                    if s["end"] > start and s["start"] < end
                ]
                # Whole overlapping cues retain original timestamps; explicitly report coverage/gaps.
                text_size, selected = 0, []
                for segment in segments:
                    if text_size + len(segment["text"]) > 12000:
                        break
                    selected.append(segment)
                    text_size += len(segment["text"])
                covered = 0
                cursor = start
                for segment in selected:
                    a, b = max(start, segment["start"]), min(end, segment["end"])
                    covered += max(0, b - max(a, cursor))
                    cursor = max(cursor, b)
                images = []
                if args.mode == "joint" and self.vision and info["has_video"]:
                    count = min(args.frame_count, image_budget)
                    for index in range(count):
                        seconds = round(start + (end - start) * (index + 0.5) / count, 3)
                        seconds = min(seconds, info["duration"] - 0.001)
                        images.append(
                            self.store.media_frame(self.space_id, args.material_id, seconds)
                        )
                return {
                    "material_id": args.material_id,
                    "name": info["name"],
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration": info["duration"],
                    "transcript_source": info["transcript_source"],
                    "transcript_state": info["transcript_state"],
                    "alignment_notes": info.get("alignment_notes", []),
                    "segments": selected,
                    "truncated": len(selected) < len(segments),
                    "caption_coverage_ratio": round(covered / (end - start), 3),
                    "images": images,
                    "frame_status": (
                        "sampled"
                        if images
                        else "budget_exhausted"
                        if args.mode == "joint" and info["has_video"] and self.vision
                        else "not_requested_or_unavailable"
                    ),
                    "source_url": f"/#media={args.material_id}&start={start}&end={end}",
                    "detail": "字幕可能存在识别错误或空白，覆盖率仅表示字幕区间覆盖；图像为标注时间附近抽样帧，不代表观看或听完连续片段。",
                }
            if name in {"read_page", "inspect_page"}:
                page = self.store.page(self.space_id, args.material_id, args.page)
                result = {
                    "material_id": args.material_id,
                    "name": page["name"],
                    "page": args.page,
                    "page_url": f"/api/spaces/{self.space_id}/materials/{args.material_id}/pages/{args.page}/image",
                }
                if name == "read_page":
                    lines = page["text"].splitlines()
                    text = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
                    return {
                        **result,
                        "text": text[:12000],
                        "total_lines": len(lines),
                        "truncated": len(text) > 12000,
                        "has_text": bool(text.strip()),
                    }
                if not self.vision:
                    return {
                        "error": "vision_disabled",
                        "detail": "当前未启用视觉输入，不能声称已查看图像",
                    }
                region = args.region
                if region is not None and not (
                    all(0 <= v <= 1 for v in region)
                    and region[2] - region[0] >= 0.02
                    and region[3] - region[1] >= 0.02
                ):
                    return {
                        "error": "invalid_region",
                        "detail": "region必须为合法归一化矩形，宽高至少0.02",
                    }
                image = self.store.page_image(self.space_id, args.material_id, args.page, region)
                return {
                    **result,
                    "region": region,
                    "image": image,
                    "detail": "对应页面图像由运行时附加，请依据实际画面判断，不能猜测不可见内容。",
                }
            if name == "read_passage":
                if args.end_line < args.start_line or args.end_line - args.start_line >= 50:
                    return {"error": "invalid_range", "detail": "请选择顺序正确的1至50行"}
                material = self.store.material(self.space_id, args.material_id)
                if material["format"] == "pdf":
                    return {
                        "error": "use_read_page",
                        "detail": "PDF请使用read_page，以保留准确物理页码",
                    }
                if material["format"] in {"audio", "video"}:
                    return {
                        "error": "use_inspect_media",
                        "detail": "音视频请使用inspect_media保留时间轴",
                    }
                lines = material["text"].splitlines()
                if args.start_line > len(lines):
                    return {"error": "invalid_range", "total_lines": len(lines)}
                end = min(args.end_line, len(lines))
                text = "\n".join(f"{i + 1}: {lines[i]}" for i in range(args.start_line - 1, end))
                return {
                    "material_id": material["id"],
                    "name": material["name"],
                    "start_line": args.start_line,
                    "end_line": end,
                    "total_lines": len(lines),
                    "text": text[:8000],
                    "truncated": len(text) > 8000,
                }
            if args.material_id:
                self.store.material(self.space_id, args.material_id)
            query = terms(args.query)
            hits = []
            for item in self.store.materials(self.space_id):
                if args.material_id and item["id"] != args.material_id:
                    continue
                material = self.store.material(self.space_id, item["id"])
                if material["format"] in {"audio", "video"}:
                    for segment in self.store.segments(self.space_id, item["id"]):
                        overlap = len(query & terms(segment["text"]))
                        if overlap:
                            hits.append(
                                {
                                    **item,
                                    "start_seconds": segment["start"],
                                    "end_seconds": segment["end"],
                                    "excerpt": segment["text"][:1000],
                                    "score": overlap,
                                    "source_url": f"/#media={item['id']}&start={segment['start']}&end={segment['end']}",
                                }
                            )
                    continue
                pages = (
                    self.store.pages(self.space_id, item["id"])
                    if material["format"] == "pdf"
                    else [{"page": None}]
                )
                for page in pages:
                    content = (
                        self.store.page(self.space_id, item["id"], page["page"])["text"]
                        if page["page"]
                        else material["text"]
                    )
                    lines = content.splitlines()
                    for start in range(0, len(lines), 12):
                        text = "\n".join(lines[start : start + 12])
                        overlap = len(query & terms(text))
                        if overlap:
                            hits.append(
                                {
                                    **item,
                                    "page": page["page"],
                                    "start_line": start + 1,
                                    "end_line": min(start + 12, len(lines)),
                                    "excerpt": text[:1000],
                                    "score": overlap,
                                }
                            )
            hits.sort(
                key=lambda item: (
                    -item["score"],
                    item["id"],
                    item.get("page") or 0,
                    item.get("start_line", item.get("start_seconds", 0)),
                )
            )
            return {"hits": hits[:5], "retrieval": "lexical_baseline"}
        except ValidationError:
            return {"error": "invalid_arguments", "detail": "请按工具参数定义重试"}
        except LookupError as error:
            return {"error": "material_not_found", "detail": str(error)}
        except PdfError as error:
            return {"error": "render_failed", "detail": str(error)}
        except MediaError as error:
            return {"error": "media_failed", "detail": str(error)}
