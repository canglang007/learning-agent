import base64
import copy
import json
from dataclasses import dataclass

import httpx

from learning_agent.config import Settings


class ModelError(Exception):
    """Sanitized provider failure; do not expose upstream bodies or credentials."""


def wire_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


@dataclass
class ModelRequest:
    endpoint: str
    payload: dict
    assets: list
    encoded: bool


def asset_info(root, attachment):
    from PIL import Image

    root = root.resolve()
    path = (root / attachment["asset_path"]).resolve()
    try:
        if not path.is_relative_to(root) or path.suffix != ".png":
            raise ValueError()
        size = path.stat().st_size
        if size > 6_000_000:
            raise ValueError()
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError()
            width, height = image.size
            image.verify()
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ModelError("页面图像产物不可用或过大，请重新查看页面或选择更小区域。") from None
    return {"asset_path": attachment["asset_path"], "bytes": size, "width": width, "height": height}


class OpenAIProvider:
    def __init__(self, settings: Settings, transport=None):
        self.settings, self.transport = settings, transport

    def content(self, message, responses=False, encode_images=True):
        images = message.get("attachments", [])
        if not images:
            return message.get("content", "")
        content = [
            {"type": "input_text" if responses else "text", "text": message.get("content", "")}
        ]
        root = self.settings.data_root.resolve()
        for attachment in images:
            asset_info(root, attachment)
            path = root / attachment["asset_path"]
            try:
                url = (
                    "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
                    if encode_images
                    else "asset-preview"
                )
            except OSError:
                raise ModelError("页面图像产物不可用，请重新查看页面。") from None
            content.append(
                {"type": "input_image", "image_url": url, "detail": "high"}
                if responses
                else {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
            )
        return content

    def build_request(self, history, instructions, tools, allow_tools=True, *, encode_images=True):
        settings = self.settings
        common = {"model": settings.model}
        if settings.protocol == "chat_completions":
            messages = []
            for item in history:
                message = {
                    k: v
                    for k, v in item.items()
                    if k in {"role", "content", "tool_calls", "tool_call_id", "name"}
                }
                message["content"] = self.content(item, encode_images=encode_images)
                messages.append(message)
            payload = {
                **common,
                "messages": [{"role": "system", "content": instructions}, *messages],
                "tools": [{"type": "function", "function": tool} for tool in tools],
                "tool_choice": "auto" if allow_tools else "none",
            }
            endpoint = "/chat/completions"
        else:
            items = []
            for message in history:
                if message.get("provider_items") is not None:
                    items.extend(message["provider_items"])
                elif message["role"] == "tool":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": message["tool_call_id"],
                            "output": message["content"],
                        }
                    )
                else:
                    items.append(
                        {
                            "role": message["role"],
                            "content": self.content(
                                message, responses=True, encode_images=encode_images
                            ),
                        }
                    )
            payload = {
                **common,
                "instructions": instructions,
                "input": items,
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "tools": [{"type": "function", **tool} for tool in tools],
                "tool_choice": "auto" if allow_tools else "none",
            }
            endpoint = "/responses"
        if settings.output_token_field != "none":
            payload[settings.output_token_field] = settings.context_output_reserve_tokens
        assets = [
            asset_info(settings.data_root, a)
            for m in history
            if not (settings.protocol == "responses" and m.get("provider_items") is not None)
            for a in m.get("attachments", [])
        ]
        return ModelRequest(endpoint, payload, assets, encode_images)

    def estimate(self, request):
        payload = copy.deepcopy(request.payload)
        opaque = 0

        def strip(value):
            nonlocal opaque
            if isinstance(value, list):
                for item in value:
                    strip(item)
            elif isinstance(value, dict):
                if value.get("type") == "reasoning":
                    opaque += len(wire_bytes(value))
                    value.clear()
                elif value.get("type") in {"image_url", "input_image"}:
                    # The image's base64 is transport, not ordinary text tokens.
                    value["image_url"] = ""
                else:
                    for item in value.values():
                        strip(item)

        strip(payload)
        text_units = len(wire_bytes(payload)) + 64
        request_bytes = len(wire_bytes(request.payload))
        if not request.encoded:
            request_bytes += sum(
                4 * ((a["bytes"] + 2) // 3) + len("data:image/png;base64,") - len("asset-preview")
                for a in request.assets
            )
        image_units = len(request.assets) * self.settings.context_image_allowance
        return {
            "estimator": "utf8-proxy-v1",
            "text_units": text_units,
            "opaque_units": opaque,
            "image_units": image_units,
            "input_total": text_units + opaque + image_units,
            "request_bytes": request_bytes,
            "images": len(request.assets),
            "uncertainty": ["tokenizer_unverified", "model_window_unverified"]
            + (["vision_allowance_uncalibrated"] if request.assets else [])
            + (["opaque_cost_unknown"] if opaque else []),
            "output_limit_enforced": self.settings.output_token_field != "none",
        }

    async def step(self, history, instructions, tools, allow_tools=True):
        return await self.send(self.build_request(history, instructions, tools, allow_tools))

    async def send(self, request):
        settings = self.settings
        if not request.encoded:
            raise ModelError("内部请求尚未完成图像编码，未发送模型请求。")
        body = wire_bytes(request.payload)
        if len(body) > settings.context_max_request_bytes:
            raise ModelError("模型请求体积过大，请缩小资料片段。")
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=settings.timeout_seconds
            ) as client:
                response = await client.post(
                    settings.base_url.rstrip("/") + request.endpoint,
                    headers={
                        "Authorization": "Bearer " + settings.api_key.get_secret_value(),
                        "Content-Type": "application/json",
                    },
                    content=body,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as error:
            raise ModelError(
                f"模型服务返回 HTTP {error.response.status_code}，请检查协议、型号或额度。"
            ) from None
        except (httpx.RequestError, ValueError):
            raise ModelError("模型服务连接失败、超时或返回了无效 JSON。") from None
        return self.parse_response(data)

    def parse_response(self, data):
        settings = self.settings
        try:
            if settings.protocol == "chat_completions":
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ModelError("模型输出被截断，本轮未完成。")
                raw = choice["message"]
                message = {"role": "assistant", "content": raw.get("content") or ""}
                if raw.get("tool_calls"):
                    message["tool_calls"] = raw["tool_calls"]
            else:
                if data.get("status") != "completed":
                    raise ModelError("模型响应未完成，本轮未完成。")
                output = data["output"]
                text = "\n".join(
                    part.get("text", part.get("refusal", ""))
                    for item in output
                    if item["type"] == "message"
                    for part in item.get("content", [])
                    if part["type"] in {"output_text", "refusal"}
                )
                calls = [
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]},
                    }
                    for item in output
                    if item["type"] == "function_call"
                ]
                message = {"role": "assistant", "content": text, "provider_items": output}
                if calls:
                    message["tool_calls"] = calls
            for call in message.get("tool_calls", []):
                if (
                    not isinstance(call["id"], str)
                    or not isinstance(call["function"]["arguments"], str)
                    or not isinstance(call["function"]["name"], str)
                ):
                    raise TypeError()
            if not isinstance(message["content"], str):
                raise TypeError()
            usage = data.get("usage", {})
            return message, usage if isinstance(usage, dict) else {}
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ModelError("模型响应不符合所选接口协议。") from None
