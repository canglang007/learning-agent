"""Protocol assembly and a deliberately narrow, provisional answer projection."""

import json


class InvalidStream(ValueError):
    pass


def string_prefix(source):
    """Decode only complete Unicode scalars from an unfinished JSON string."""
    result, i = [], 1
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while i < len(source):
        char = source[i]
        if char == '"':
            break
        if char == "\\":
            if i + 1 >= len(source):
                break
            code = source[i + 1]
            if code == "u":
                if i + 6 > len(source):
                    break
                try:
                    value = int(source[i + 2 : i + 6], 16)
                    i += 6
                    if 0xD800 <= value <= 0xDBFF:
                        if i + 6 > len(source):
                            break
                        if source[i : i + 2] != "\\u":
                            raise InvalidStream()
                        low = int(source[i + 2 : i + 6], 16)
                        if not 0xDC00 <= low <= 0xDFFF:
                            raise InvalidStream()
                        value = 0x10000 + ((value - 0xD800) << 10) + low - 0xDC00
                        i += 6
                    elif 0xDC00 <= value <= 0xDFFF:
                        raise InvalidStream()
                    char = chr(value)
                except ValueError:
                    raise InvalidStream() from None
            elif code in escapes:
                char = escapes[code]
                i += 2
            else:
                raise InvalidStream()
        else:
            if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
                raise InvalidStream()
            i += 1
        result.append(char)
    return "".join(result)


def answer_prefix(arguments):
    """Only the root object's answer string; never nested keys or arbitrary JSON."""
    decoder, i = json.JSONDecoder(), 0
    source = arguments.lstrip()
    if not source.startswith("{"):
        return ""
    i = 1
    while i < len(source):
        while i < len(source) and source[i].isspace():
            i += 1
        try:
            key, end = decoder.raw_decode(source, i)
        except ValueError:
            return ""
        if not isinstance(key, str):
            return ""
        i = end
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source) or source[i] != ":":
            return ""
        i += 1
        while i < len(source) and source[i].isspace():
            i += 1
        if key == "answer":
            return string_prefix(source[i:]) if source[i : i + 1] == '"' else ""
        try:
            _, i = decoder.raw_decode(source, i)
        except ValueError:
            return ""
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source) or source[i] != ",":
            return ""
        i += 1
    return ""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidStream("duplicate key")
        result[key] = value
    return result


class StreamAssembly:
    """Keep protocol state private; return only answer snapshots to the draft sink."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.calls, self.content, self.usage = {}, "", {}
        self.finished, self.done, self.response = None, False, None

    def feed(self, data):
        if not isinstance(data, dict) or data.get("error"):
            raise InvalidStream()
        if self.protocol == "chat_completions":
            if self.finished and data.get("choices"):
                raise InvalidStream()
            choices = data.get("choices", [])
            for choice in choices:
                if choice.get("index", 0) != 0:
                    raise InvalidStream()
                delta = choice.get("delta", {})
                self.content += delta.get("content") or ""
                for part in delta.get("tool_calls", []):
                    index = part["index"]
                    if not isinstance(index, int) or not 0 <= index < 32:
                        raise InvalidStream()
                    call = self.calls.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    call["id"] += part.get("id") or ""
                    for key in ("name", "arguments"):
                        call["function"][key] += part.get("function", {}).get(key) or ""
                if choice.get("finish_reason"):
                    self.finished = choice["finish_reason"]
            if isinstance(data.get("usage"), dict):
                self.usage = data["usage"]
        else:
            kind = data.get("type")
            if self.response is not None:
                raise InvalidStream()
            if kind in {"response.failed", "response.incomplete", "error"}:
                raise InvalidStream()
            if kind == "response.output_item.added" and data["item"]["type"] == "function_call":
                item = data["item"]
                self.calls[data["output_index"]] = {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item.get("arguments", "")},
                }
            elif kind == "response.function_call_arguments.delta":
                self.calls[data["output_index"]]["function"]["arguments"] += data["delta"]
            elif kind == "response.output_text.delta":
                self.content += data["delta"]
            elif kind == "response.completed":
                self.response = data["response"]
                self.done = True

    def projection(self):
        if self.calls:
            if len(self.calls) != 1:
                return ""
            function = next(iter(self.calls.values()))["function"]
            if function["name"] != "deliver_explanation":
                return ""
            return answer_prefix(function["arguments"])[:16000]
        return self.content[:16000]

    def result(self):
        if self.protocol == "chat_completions":
            if not self.done or self.finished not in {"stop", "tool_calls"}:
                raise InvalidStream()
            raw = {"role": "assistant", "content": self.content}
            if self.calls:
                raw["tool_calls"] = [self.calls[k] for k in sorted(self.calls)]
                if any(not c["id"] or not c["function"]["name"] for c in raw["tool_calls"]):
                    raise InvalidStream()
            return {
                "choices": [{"finish_reason": self.finished, "message": raw}],
                "usage": self.usage,
            }
        if self.response is None or not self.done:
            raise InvalidStream()
        return self.response


def final_projection(message):
    calls = message.get("tool_calls", [])
    if calls:
        if len(calls) != 1 or calls[0]["function"]["name"] != "deliver_explanation":
            return ""
        try:
            args = json.loads(calls[0]["function"]["arguments"], object_pairs_hook=unique_object)
            answer = args.get("answer", "") if isinstance(args, dict) else ""
            return answer if isinstance(answer, str) and len(answer) <= 16000 else ""
        except (ValueError, TypeError):
            return ""
    return message.get("content", "")[:16000]
