import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from learning_agent.store import Store


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ListArgs(Args):
    pass


class SearchArgs(Args):
    query: str = Field(min_length=1, max_length=300)


class ReadArgs(Args):
    material_id: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


ARGUMENTS = {"list_materials": ListArgs, "search_materials": SearchArgs, "read_passage": ReadArgs}
DESCRIPTIONS = {
    "list_materials": "查看当前空间可阅读的文字资料目录。",
    "search_materials": "按关键词寻找资料段落。词法候选可能不完整；无结果可改写查询或查目录阅读。",
    "read_passage": "读取资料原文与行号。范围最多50行；获取推导条件、定义或前后文。",
}


def definitions():
    result = []
    for name, model in ARGUMENTS.items():
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
    def __init__(self, store: Store, space_id: str):
        self.store, self.space_id = store, space_id

    def execute(self, name, arguments):
        if name not in ARGUMENTS:
            return {"error": "unknown_tool", "detail": "工具不存在"}
        try:
            args = ARGUMENTS[name].model_validate(arguments)
            if name == "list_materials":
                return {"materials": self.store.materials(self.space_id)}
            if name == "read_passage":
                if args.end_line < args.start_line or args.end_line - args.start_line >= 50:
                    return {"error": "invalid_range", "detail": "请选择顺序正确的1至50行"}
                material = self.store.material(self.space_id, args.material_id)
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
            query = terms(args.query)
            hits = []
            for item in self.store.materials(self.space_id):
                material = self.store.material(self.space_id, item["id"])
                lines = material["text"].splitlines()
                for start in range(0, len(lines), 12):
                    text = "\n".join(lines[start : start + 12])
                    overlap = len(query & terms(text))
                    if overlap:
                        hits.append(
                            {
                                **item,
                                "start_line": start + 1,
                                "end_line": min(start + 12, len(lines)),
                                "excerpt": text[:1000],
                                "score": overlap,
                            }
                        )
            hits.sort(key=lambda item: (-item["score"], item["id"], item["start_line"]))
            return {"hits": hits[:5], "retrieval": "lexical_baseline"}
        except ValidationError:
            return {"error": "invalid_arguments", "detail": "请按工具参数定义重试"}
        except LookupError as error:
            return {"error": "material_not_found", "detail": str(error)}
